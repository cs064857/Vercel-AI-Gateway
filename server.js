import express from 'express';
import { generateText, streamText } from 'ai';
import { createGateway } from '@ai-sdk/gateway';
import { v4 as uuidv4 } from 'uuid';

const app = express();
const PORT = process.env.PORT || 48000;

//最大請求體 50MB（圖片 base64 可能很大）
app.use(express.json({ limit: '50mb' }));

//圖片生成相關模型名稱（匹配判斷用）
const IMAGE_MODELS = [
  'gemini-3-pro-image',
  'gemini-3-pro-image-preview',
  'gemini-2.5-flash-image-preview',
  'gemini-2.5-flash-image',
];

//判斷是否為圖片生成模型
function isImageModel(model) {
  const modelLower = model.toLowerCase().replace('google/', '');
  return IMAGE_MODELS.some(m => modelLower.includes(m));
}

//從請求中提取 API Key
function extractApiKey(req) {
  const authHeader = req.headers['authorization'];
  if (!authHeader) return null;
  if (authHeader.startsWith('Bearer ')) {
    return authHeader.slice(7);
  }
  return authHeader;
}

//規範化模型名稱（確保帶 google/ 前綴）
function normalizeModelName(model) {
  if (model.startsWith('google/')) return model;
  return `google/${model}`;
}

//解析圖片生成參數
function parseImageParams(body) {
  //默認值
  let imageSize = '4K';
  let aspectRatio = '1:1';

  //從 body 頂層讀取（自定義擴展欄位）
  if (body.imageSize) imageSize = body.imageSize;
  if (body.resolution) imageSize = body.resolution;
  if (body.aspectRatio) aspectRatio = body.aspectRatio;
  if (body.aspect_ratio) aspectRatio = body.aspect_ratio;

  //從 providerOptions 讀取（覆蓋優先）
  const googleOpts = body.providerOptions?.google?.imageConfig;
  if (googleOpts?.imageSize) imageSize = googleOpts.imageSize;
  if (googleOpts?.aspectRatio) aspectRatio = googleOpts.aspectRatio;

  return { imageSize, aspectRatio };
}

//從 messages 中提取 prompt
function extractPrompt(messages) {
  if (!messages || messages.length === 0) return '';

  //取最後一條用戶訊息
  const lastUserMsg = [...messages].reverse().find(m => m.role === 'user');
  if (!lastUserMsg) return '';

  if (typeof lastUserMsg.content === 'string') {
    return lastUserMsg.content;
  }

  //多模態內容，提取文字部分
  if (Array.isArray(lastUserMsg.content)) {
    return lastUserMsg.content
      .filter(part => part.type === 'text')
      .map(part => part.text)
      .join('\n');
  }

  return String(lastUserMsg.content);
}

//將 OpenAI messages 轉換為 AI SDK 的 messages 格式
function convertMessages(messages) {
  return messages.map(msg => {
    if (typeof msg.content === 'string') {
      return { role: msg.role, content: msg.content };
    }

    //處理多模態消息
    if (Array.isArray(msg.content)) {
      const parts = msg.content.map(part => {
        if (part.type === 'text') {
          return { type: 'text', text: part.text };
        }
        if (part.type === 'image_url') {
          const url = typeof part.image_url === 'string'
            ? part.image_url
            : part.image_url?.url;
          if (url?.startsWith('data:')) {
            //base64 圖片
            const match = url.match(/^data:([^;]+);base64,(.+)$/);
            if (match) {
              return {
                type: 'image',
                image: match[2],
                mediaType: match[1],
              };
            }
          }
          return { type: 'image', image: new URL(url) };
        }
        return part;
      });
      return { role: msg.role, content: parts };
    }

    return { role: msg.role, content: String(msg.content) };
  });
}

//構建 OpenAI 格式的 chat completion 回應
function buildChatCompletionResponse(model, content, files = []) {
  let responseContent = content || '';

  //如果有圖片文件，以 Markdown base64 格式嵌入
  if (files && files.length > 0) {
    const imageMarkdowns = files
      .filter(f => f.mediaType?.startsWith('image/'))
      .map((f, i) => {
        const base64Data = typeof f.base64 === 'string' ? f.base64 : Buffer.from(f.uint8Array).toString('base64');
        //去除 data URL 前綴（如果有的話）
        const cleanBase64 = base64Data.replace(/^data:[^;]+;base64,/, '');
        return `![generated_image_${i + 1}](data:${f.mediaType};base64,${cleanBase64})`;
      });

    if (imageMarkdowns.length > 0) {
      responseContent = responseContent
        ? `${responseContent}\n\n${imageMarkdowns.join('\n\n')}`
        : imageMarkdowns.join('\n\n');
    }
  }

  return {
    id: `chatcmpl-${uuidv4()}`,
    object: 'chat.completion',
    created: Math.floor(Date.now() / 1000),
    model: model,
    choices: [{
      index: 0,
      message: {
        role: 'assistant',
        content: responseContent,
      },
      finish_reason: 'stop',
    }],
    usage: {
      prompt_tokens: 0,
      completion_tokens: 0,
      total_tokens: 0,
    },
  };
}

//構建 SSE 串流格式的 chunk
function buildStreamChunk(model, content, finishReason = null) {
  return {
    id: `chatcmpl-${uuidv4()}`,
    object: 'chat.completion.chunk',
    created: Math.floor(Date.now() / 1000),
    model: model,
    choices: [{
      index: 0,
      delta: finishReason ? {} : { content },
      finish_reason: finishReason,
    }],
  };
}

//處理圖片生成請求
async function handleImageGeneration(req, res) {
  const apiKey = extractApiKey(req);
  if (!apiKey) {
    return res.status(401).json({
      error: { message: 'Missing Authorization header', type: 'authentication_error' }
    });
  }

  const { model, messages, stream } = req.body;
  const { imageSize, aspectRatio } = parseImageParams(req.body);
  const prompt = extractPrompt(messages);

  if (!prompt) {
    return res.status(400).json({
      error: { message: 'No prompt found in messages', type: 'invalid_request_error' }
    });
  }

  const gatewayModel = normalizeModelName(model);

  console.log(`[圖片生成] 模型: ${gatewayModel}, 解析度: ${imageSize}, 長寬比: ${aspectRatio}`);
  console.log(`[圖片生成] Prompt: ${prompt.substring(0, 100)}...`);

  try {
    const gateway = createGateway({ apiKey });

    //使用 generateText（Gemini 圖片模型是語言模型，透過 files 返回圖片）
    const result = await generateText({
      model: gateway(gatewayModel),
      messages: convertMessages(messages),
      providerOptions: {
        google: {
          responseModalities: ['TEXT', 'IMAGE'],
          imageConfig: {
            imageSize,
            aspectRatio,
          },
        },
      },
    });

    const responseModel = model;
    const textContent = result.text || '';
    const files = result.files || [];

    if (stream) {
      //串流模式：圖片生成完成後一次性以 SSE 格式發送
      res.setHeader('Content-Type', 'text/event-stream');
      res.setHeader('Cache-Control', 'no-cache');
      res.setHeader('Connection', 'keep-alive');

      const fullResponse = buildChatCompletionResponse(responseModel, textContent, files);
      const content = fullResponse.choices[0].message.content;

      //發送內容 chunk
      const chunk = buildStreamChunk(responseModel, content);
      res.write(`data: ${JSON.stringify(chunk)}\n\n`);

      //發送結束 chunk
      const endChunk = buildStreamChunk(responseModel, null, 'stop');
      res.write(`data: ${JSON.stringify(endChunk)}\n\n`);
      res.write('data: [DONE]\n\n');
      res.end();
    } else {
      //非串流模式
      const response = buildChatCompletionResponse(responseModel, textContent, files);
      res.json(response);
    }

    console.log(`[圖片生成] 完成，生成 ${files.length} 張圖片`);
  } catch (error) {
    console.error('[圖片生成] 錯誤:', error);
    res.status(500).json({
      error: {
        message: error.message || 'Image generation failed',
        type: 'server_error',
        details: error.cause?.message || undefined,
      }
    });
  }
}

//處理通用對話請求
async function handleChatCompletion(req, res) {
  const apiKey = extractApiKey(req);
  if (!apiKey) {
    return res.status(401).json({
      error: { message: 'Missing Authorization header', type: 'authentication_error' }
    });
  }

  const { model, messages, stream } = req.body;
  const gatewayModel = normalizeModelName(model);

  console.log(`[對話] 模型: ${gatewayModel}, 串流: ${!!stream}`);

  try {
    const gateway = createGateway({ apiKey });
    const sdkMessages = convertMessages(messages);

    if (stream) {
      //串流模式
      res.setHeader('Content-Type', 'text/event-stream');
      res.setHeader('Cache-Control', 'no-cache');
      res.setHeader('Connection', 'keep-alive');

      const result = streamText({
        model: gateway(gatewayModel),
        messages: sdkMessages,
      });

      for await (const chunk of result.textStream) {
        const sseChunk = buildStreamChunk(model, chunk);
        res.write(`data: ${JSON.stringify(sseChunk)}\n\n`);
      }

      const endChunk = buildStreamChunk(model, null, 'stop');
      res.write(`data: ${JSON.stringify(endChunk)}\n\n`);
      res.write('data: [DONE]\n\n');
      res.end();
    } else {
      //非串流模式
      const result = await generateText({
        model: gateway(gatewayModel),
        messages: sdkMessages,
      });

      const response = buildChatCompletionResponse(model, result.text, result.files);
      res.json(response);
    }
  } catch (error) {
    console.error('[對話] 錯誤:', error);
    res.status(500).json({
      error: {
        message: error.message || 'Chat completion failed',
        type: 'server_error',
      }
    });
  }
}

// === 路由 ===

//健康檢查
app.get('/health', (req, res) => {
  res.json({ status: 'ok', timestamp: new Date().toISOString() });
});

//OpenAI 相容端點
app.post('/v1/chat/completions', async (req, res) => {
  const { model } = req.body;

  if (!model) {
    return res.status(400).json({
      error: { message: 'model is required', type: 'invalid_request_error' }
    });
  }

  if (isImageModel(model)) {
    return handleImageGeneration(req, res);
  }

  return handleChatCompletion(req, res);
});

//模型列表端點（基本相容）
app.get('/v1/models', (req, res) => {
  const models = IMAGE_MODELS.map(id => ({
    id: `google/${id}`,
    object: 'model',
    created: Math.floor(Date.now() / 1000),
    owned_by: 'google',
  }));
  res.json({ object: 'list', data: models });
});

//啟動伺服器
app.listen(PORT, '0.0.0.0', () => {
  console.log(`=== Vercel AI Gateway Proxy ===`);
  console.log(`監聽端口: ${PORT}`);
  console.log(`端點: POST /v1/chat/completions`);
  console.log(`健康檢查: GET /health`);
  console.log(`默認圖片解析度: 4K`);
  console.log(`支援模型: ${IMAGE_MODELS.map(m => `google/${m}`).join(', ')}`);
  console.log(`==============================`);
});
