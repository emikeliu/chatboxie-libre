import type { ProviderModelInfo } from '@shared/types'
import type { ModelDependencies } from '@shared/types/adapters'
import { getReasoningProviderOptions, OPENAI_REASONING_EFFORT_LEVELS } from '@shared/utils/reasoning-control'
import type { SentryScope } from '@shared/utils/sentry_adapter'
import { describe, expect, it, vi } from 'vitest'
import CustomOpenAI from './custom-openai'

const PROVIDER_ID = 'custom-provider-acme'

function createDependencies() {
  return {
    request: {
      apiRequest: vi.fn(),
      fetchWithOptions: vi.fn(),
    },
    storage: {
      saveImage: vi.fn(),
      getImage: vi.fn(),
    },
    sentry: {
      captureException: vi.fn(),
      withScope: vi.fn((callback: (scope: SentryScope) => void) =>
        callback({
          setTag: vi.fn(),
          setExtra: vi.fn(),
        })
      ),
    },
    getRemoteConfig: vi.fn(),
    platformType: 'desktop',
  } as unknown as ModelDependencies & {
    request: { apiRequest: ReturnType<typeof vi.fn>; fetchWithOptions: ReturnType<typeof vi.fn> }
  }
}

function chatCompletionStream(): Response {
  const chunks = [
    {
      id: 'chat-1',
      object: 'chat.completion.chunk',
      created: 1,
      model: 'acme-reasoner-v2',
      choices: [{ index: 0, delta: { role: 'assistant', content: 'ok' }, finish_reason: null }],
    },
    {
      id: 'chat-1',
      object: 'chat.completion.chunk',
      created: 1,
      model: 'acme-reasoner-v2',
      choices: [{ index: 0, delta: {}, finish_reason: 'stop' }],
      usage: { prompt_tokens: 2, completion_tokens: 1, total_tokens: 3 },
    },
  ]
  return new Response(`${chunks.map((chunk) => `data: ${JSON.stringify(chunk)}\n\n`).join('')}data: [DONE]\n\n`, {
    headers: { 'content-type': 'text/event-stream' },
  })
}

function createModel(model: ProviderModelInfo, dependencies: ModelDependencies) {
  return new CustomOpenAI(
    {
      apiKey: 'test-key',
      apiHost: 'https://acme.example.com/v1',
      apiPath: '/chat/completions',
      model,
      stream: true,
    },
    dependencies
  )
}

const declaredReasoningModel: ProviderModelInfo = {
  modelId: 'acme-reasoner-v2',
  type: 'chat',
  apiStyle: 'openai',
  providerId: PROVIDER_ID,
  capabilities: ['reasoning'],
}

describe('generic OpenAI-compatible reasoning effort', () => {
  // Every effort value an OpenAI-compatible endpoint may expose is forwarded verbatim.
  it.each(OPENAI_REASONING_EFFORT_LEVELS)(
    'sends reasoning_effort=%s for a model declared as reasoning',
    async (level) => {
      const dependencies = createDependencies()
      dependencies.request.apiRequest.mockResolvedValueOnce(chatCompletionStream())
      const model = createModel(declaredReasoningModel, dependencies)

      await model.chat([{ role: 'user', content: 'Hello' }], {
        sessionId: 'generic-reasoning-session',
        providerOptions: getReasoningProviderOptions(PROVIDER_ID, declaredReasoningModel, level),
      })

      const request = dependencies.request.apiRequest.mock.calls[0][0]
      const body = JSON.parse(String(request.body))
      expect(body.reasoning_effort).toBe(level)
      expect(request.url).toBe('https://acme.example.com/v1/chat/completions')
    }
  )

  it.each(['off', 'default'] as const)('omits reasoning_effort for the %s level', async (level) => {
    const dependencies = createDependencies()
    dependencies.request.apiRequest.mockResolvedValueOnce(chatCompletionStream())
    const model = createModel(declaredReasoningModel, dependencies)

    await model.chat([{ role: 'user', content: 'Hello' }], {
      sessionId: 'generic-reasoning-session',
      providerOptions: getReasoningProviderOptions(PROVIDER_ID, declaredReasoningModel, level),
    })

    const body = JSON.parse(String(dependencies.request.apiRequest.mock.calls[0][0].body))
    expect(body).not.toHaveProperty('reasoning_effort')
  })

  it('forwards reasoning_effort for any chat model on the compatible endpoint', async () => {
    const dependencies = createDependencies()
    dependencies.request.apiRequest.mockResolvedValueOnce(chatCompletionStream())
    const plainChatModel: ProviderModelInfo = {
      modelId: 'acme-chat-v2',
      type: 'chat',
      apiStyle: 'openai',
      providerId: PROVIDER_ID,
      capabilities: ['vision'],
    }
    const model = createModel(plainChatModel, dependencies)

    await model.chat([{ role: 'user', content: 'Hello' }], {
      sessionId: 'plain-chat-session',
      providerOptions: getReasoningProviderOptions(PROVIDER_ID, plainChatModel, 'high'),
    })

    const body = JSON.parse(String(dependencies.request.apiRequest.mock.calls[0][0].body))
    expect(body.reasoning_effort).toBe('high')
  })
})
