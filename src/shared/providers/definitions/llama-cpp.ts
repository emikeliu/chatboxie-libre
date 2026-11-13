import { ModelProviderEnum, ModelProviderType } from '../../types'
import { defineProvider } from '../registry'
import CustomOpenAI from './models/custom-openai'

/**
 * llama.cpp's built-in HTTP server: an OpenAI-compatible Chat Completions endpoint.
 * The bundled defaults point at the standard local port (9931) and pin a single
 * "local" chat model so a fresh install can talk to a locally running llama.cpp
 * without any configuration.
 */
export const llamaCppProvider = defineProvider({
  id: ModelProviderEnum.LlamaCpp,
  name: 'llama.cpp',
  type: ModelProviderType.OpenAI,
  defaultSettings: {
    apiHost: 'http://127.0.0.1:9931',
    apiPath: '/v1/chat/completions',
    apiKey: 'chageme',
    models: [
      {
        modelId: 'local',
        nickname: '本地模型',
        type: 'chat',
        capabilities: ['vision', 'reasoning', 'tool_use'],
        contextWindow: 1_000_000,
        maxOutput: 262_144,
      },
    ],
  },
  createModel: (config) => {
    return new CustomOpenAI(
      {
        apiKey: config.effectiveApiKey,
        apiHost: config.formattedApiHost,
        apiPath: config.formattedApiPath,
        model: config.model,
        temperature: config.settings.temperature,
        topP: config.settings.topP,
        maxOutputTokens: config.settings.maxTokens,
        stream: config.settings.stream,
        useProxy: config.providerSetting.useProxy,
      },
      config.dependencies
    )
  },
  getDisplayName: (modelId, providerSettings) => {
    const nickname = providerSettings?.models?.find((model) => model.modelId === modelId)?.nickname
    return nickname || modelId
  },
})
