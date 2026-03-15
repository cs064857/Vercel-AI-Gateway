import express from 'express';
import { generateText, streamText } from 'ai';
import { createGateway } from '@ai-sdk/gateway';
import { v4 as uuidv4 } from 'uuid';

const app = express();
const PORT = process.env.PORT || 48000;

//最大請求體 50MB（圖片 base64 可能很大）
app.use(express.json({ limit: '50mb' }));

// 請求與響應日誌記錄
app.use((req, res, next) => {
  console.log(`\n=== 收到請求: ${req.method} ${req.originalUrl} ===`);
  console.log(`[請求頭]`, JSON.stringify(req.headers, null, 2));
  
  if (req.body && Object.keys(req.body).length > 0) {
    try {
      const logBody = JSON.parse(JSON.stringify(req.body));
      if (Array.isArray(logBody.messages)) {
        logBody.messages.forEach(m => {
          if (Array.isArray(m.content)) {
            m.content.forEach(p => {
              if (p.type === 'image_url' && p.image_url) {
                if (typeof p.image_url === 'string' && p.image_url.length > 200) {
                  p.image_url = p.image_url.substring(0, 50) + '...[TRUNCATED_IMAGE_DATA]';
                } else if (typeof p.image_url.url === 'string' && p.image_url.url.length > 200) {
                  p.image_url.url = p.image_url.url.substring(0, 50) + '...[TRUNCATED_IMAGE_DATA]';
                }
              }
            });
          }
        });
      }
      console.log(`[請求參數]`, JSON.stringify(logBody, null, 2));
    } catch (e) {
      console.log(`[請求參數] 日誌轉化失敗`);
    }
  }

  res.on('finish', () => {
    console.log(`=== 請求結束: ${req.method} ${req.originalUrl} [HTTP ${res.statusCode}] ===`);
    console.log(`[響應頭]`, JSON.stringify(res.getHeaders(), null, 2));
  });

  next();
});

//圖片生成相關模型名稱（匹配判斷用）
const IMAGE_MODELS = [
  'gemini-3-pro-image',
  'gemini-3-pro-image-preview',
  'gemini-2.5-flash-image-preview',
  'gemini-2.5-flash-image',
  'gemini-3.1-flash-image-preview',
];

//判斷是否為圖片生成模型
function isImageModel(model) {
  if (!model) return false;
  const modelLower = model.toLowerCase().replace('google/', '');
  if (modelLower.includes('gemini') && modelLower.includes('image')) {
    return true;
  }
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

//規範化模型名稱（如果沒有提供商前綴，預設使用 google/）
function normalizeModelName(model) {
  if (model.includes('/')) return model;
  return `google/${model}`;
}

//解析圖片生成參數
function parseImageParams(body) {
  // 默認值
  const config = {
    imageSize: '4K',
    aspectRatio: '16:9'
  };

  // 提取標準欄位外的所有自定義參數
  const standardKeys = ['model', 'messages', 'stream', 'input', 'instructions'];
  const extraBodyParams = {};
  for (const key of Object.keys(body)) {
    if (!standardKeys.includes(key) && key !== 'providerOptions') {
      extraBodyParams[key] = body[key];
    }
  }

  // 相容舊的命名
  if (extraBodyParams.resolution) {
    extraBodyParams.imageSize = extraBodyParams.resolution;
    delete extraBodyParams.resolution;
  }
  if (extraBodyParams.aspect_ratio) {
    extraBodyParams.aspectRatio = extraBodyParams.aspect_ratio;
    delete extraBodyParams.aspect_ratio;
  }

  // 從 providerOptions 讀取
  const googleOpts = body.providerOptions?.google?.imageConfig || {};

  // 合併並以使用者傳的參數為優先覆蓋
  return { ...config, ...extraBodyParams, ...googleOpts };
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

//將請求體正規化為 AI SDK 的 messages 格式
function normalizeToSdkMessages(body, format) {
  if (format === 'chat') {
    return convertMessages(body.messages || []);
  }

  if (format === 'responses') {
    const sdkMessages = [];
    if (body.instructions) {
      sdkMessages.push({ role: 'system', content: body.instructions });
    }

    const inputItems = body.input || [];
    for (const item of inputItems) {
      if (item.type === 'message') {
        if (typeof item.content === 'string') {
          sdkMessages.push({ role: item.role || 'user', content: item.content });
        } else if (Array.isArray(item.content)) {
          const converted = convertMessages([{ role: item.role || 'user', content: item.content }]);
          sdkMessages.push(converted[0]);
        }
      } else if (item.type === 'input_text') {
        sdkMessages.push({ role: 'user', content: item.text });
      } else if (item.type === 'input_image') {
        const url = typeof item.image_url === 'string' ? item.image_url : item.image_url?.url;
        let imagePart;
        if (url?.startsWith('data:')) {
          const match = url.match(/^data:([^;]+);base64,(.+)$/);
          if (match) {
            imagePart = { type: 'image', image: match[2], mediaType: match[1] };
          }
        }
        if (!imagePart && url) {
          imagePart = { type: 'image', image: new URL(url) };
        }
        if (imagePart) {
          sdkMessages.push({ role: 'user', content: [imagePart] });
        }
      }
    }

    // 將相鄰同 role 的訊息合併，以避免部分模型不支援連續 user message 的錯誤
    let merged = [];
    for (const msg of sdkMessages) {
      if (merged.length > 0 && merged[merged.length - 1].role === msg.role) {
        let prevContent = merged[merged.length - 1].content;
        let currContent = msg.content;
        if (!Array.isArray(prevContent)) prevContent = [{ type: 'text', text: prevContent }];
        if (!Array.isArray(currContent)) currContent = [{ type: 'text', text: currContent }];
        merged[merged.length - 1].content = [...prevContent, ...currContent];
      } else {
        merged.push(msg);
      }
    }
    return merged;
  }
  return [];
}

//構建 API 回應
function buildAPIResponse(format, model, content, files = []) {
  let responseContent = content || '';

  //如果有圖片文件，以 Markdown base64 格式嵌入
  if (files && files.length > 0) {
    const imageMarkdowns = files
      .filter(f => f.mediaType?.startsWith('image/'))
      .map((f, i) => {
        const base64Data = typeof f.base64 === 'string' ? f.base64 : Buffer.from(f.uint8Array).toString('base64');
        //去除 data URL 前綴
        const cleanBase64 = base64Data.replace(/^data:[^;]+;base64,/, '');
        return `![generated_image_${i + 1}](data:${f.mediaType};base64,${cleanBase64})`;
      });

    if (imageMarkdowns.length > 0) {
      responseContent = responseContent
        ? `${responseContent}\n\n${imageMarkdowns.join('\n\n')}`
        : imageMarkdowns.join('\n\n');
    }
  }

  if (format === 'responses') {
    return {
      id: `resp-${uuidv4()}`,
      object: 'response',
      created: Math.floor(Date.now() / 1000),
      model: model,
      status: 'completed',
      output: [{
        type: 'message',
        role: 'assistant',
        content: responseContent,
      }],
      usage: {
        prompt_tokens: 0,
        completion_tokens: 0,
        total_tokens: 0,
      },
    };
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
function buildStreamChunk(format, model, content, finishReason = null) {
  if (format === 'responses') {
    return {
      id: `resp-${uuidv4()}`,
      object: 'response.chunk',
      created: Math.floor(Date.now() / 1000),
      model: model,
      output: [{
        type: 'message',
        role: 'assistant',
        content: content || ''
      }]
    };
  }

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
async function handleImageGeneration(req, res, format) {
  const apiKey = extractApiKey(req);
  if (!apiKey) {
    return res.status(401).json({
      error: { message: 'Missing Authorization header', type: 'authentication_error' }
    });
  }

  const { model, stream } = req.body;
  const imageConfig = parseImageParams(req.body);
  const sdkMessages = normalizeToSdkMessages(req.body, format);
  const prompt = extractPrompt(sdkMessages);

  if (!prompt) {
    return res.status(400).json({
      error: { message: 'No prompt found in request', type: 'invalid_request_error' }
    });
  }

  const gatewayModel = normalizeModelName(model);

  console.log(`[圖片生成] 模型: ${gatewayModel}, format: ${format}, imageConfig: ${JSON.stringify(imageConfig)}`);
  console.log(`[圖片生成] Prompt: ${prompt.substring(0, 100)}...`);

  try {
    const gateway = createGateway({ apiKey });

    //使用 generateText（Gemini 圖片模型是語言模型，透過 files 返回圖片）
    const result = await generateText({
      model: gateway(gatewayModel),
      messages: sdkMessages,
      providerOptions: {
        google: {
          responseModalities: ['TEXT', 'IMAGE'],
          imageConfig: imageConfig,
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

      const fullResponse = buildAPIResponse(format, responseModel, textContent, files);
      
      let chunkContent = textContent;
      if (format === 'chat') {
        chunkContent = fullResponse.choices[0].message.content;
      } else if (format === 'responses') {
        chunkContent = fullResponse.output[0].content;
      }

      //發送內容 chunk
      const chunk = buildStreamChunk(format, responseModel, chunkContent);
      res.write(`data: ${JSON.stringify(chunk)}\n\n`);

      //發送結束 chunk
      const endChunk = buildStreamChunk(format, responseModel, null, 'stop');
      res.write(`data: ${JSON.stringify(endChunk)}\n\n`);
      res.write('data: [DONE]\n\n');
      res.end();
    } else {
      //非串流模式
      const response = buildAPIResponse(format, responseModel, textContent, files);
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
async function handleChatCompletion(req, res, format) {
  const apiKey = extractApiKey(req);
  if (!apiKey) {
    return res.status(401).json({
      error: { message: 'Missing Authorization header', type: 'authentication_error' }
    });
  }

  const { model, stream } = req.body;
  const gatewayModel = normalizeModelName(model);

  console.log(`[對話] 模型: ${gatewayModel}, format: ${format}, 串流: ${!!stream}`);

  try {
    const gateway = createGateway({ apiKey });
    const sdkMessages = normalizeToSdkMessages(req.body, format);

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
        const sseChunk = buildStreamChunk(format, model, chunk);
        res.write(`data: ${JSON.stringify(sseChunk)}\n\n`);
      }

      const endChunk = buildStreamChunk(format, model, null, 'stop');
      res.write(`data: ${JSON.stringify(endChunk)}\n\n`);
      res.write('data: [DONE]\n\n');
      res.end();
    } else {
      //非串流模式
      const result = await generateText({
        model: gateway(gatewayModel),
        messages: sdkMessages,
      });

      const response = buildAPIResponse(format, model, result.text, result.files);
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
    return handleImageGeneration(req, res, 'chat');
  }

  return handleChatCompletion(req, res, 'chat');
});

//OpenAI 相容端點 - 額外支援 /v1/responses
app.post('/v1/responses', async (req, res) => {
  const { model } = req.body;

  if (!model) {
    return res.status(400).json({
      error: { message: 'model is required', type: 'invalid_request_error' }
    });
  }

  if (isImageModel(model)) {
    return handleImageGeneration(req, res, 'responses');
  }

  return handleChatCompletion(req, res, 'responses');
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
  console.log(`端點: POST /v1/chat/completions, POST /v1/responses`);
  console.log(`健康檢查: GET /health`);
  console.log(`默認圖片解析度: 4K`);
  console.log(`支援模型: ${IMAGE_MODELS.map(m => `google/${m}`).join(', ')}`);
  console.log(`==============================`);
});
