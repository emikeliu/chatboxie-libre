import { settings as getDefaultSettings, newConfigs } from 'src/shared/defaults'
import { getModel } from 'src/shared/providers'
import OpenAI from 'src/shared/providers/definitions/models/openai'
import OpenAIResponses from 'src/shared/providers/definitions/models/openai-responses'
import { ModelProviderEnum, type SessionSettings, type Settings } from 'src/shared/types'
import type { ModelDependencies } from 'src/shared/types/adapters'
import type { SentryScope } from 'src/shared/utils/sentry_adapter'
import { describe, expect, it, vi } from 'vitest'

const mockScope: SentryScope = {
  setTag: vi.fn(),
  setExtra: vi.fn(),
}

const mockDependencies: ModelDependencies = {
  request: {
    fetchWithOptions: vi.fn(),
    apiRequest: vi.fn(),
  },
  storage: {
    saveImage: vi.fn(),
    getImage: vi.fn(),
  },
  sentry: {
    captureException: vi.fn(),
    withScope: vi.fn((callback: (scope: SentryScope) => void) => callback(mockScope)),
  },
  getRemoteConfig: vi.fn(),
  platformType: 'desktop',
  oauth: {
    refreshCredential: vi.fn(),
    persistCredential: vi.fn(),
    clearCredential: vi.fn(),
  },
}

describe('getModel', () => {
  it('returns OpenAIResponses when OpenAI uses OAuth mode', () => {
    const sessionSettings: SessionSettings = {
      provider: ModelProviderEnum.OpenAI,
      modelId: 'gpt-5',
      temperature: 0.7,
      topP: 0.9,
      maxTokens: 2048,
      stream: true,
    }

    const defaultSettings = getDefaultSettings()
    const globalSettings: Settings = {
      ...defaultSettings,
      providers: {
        ...defaultSettings.providers,
        [ModelProviderEnum.OpenAI]: {
          activeAuthMode: 'oauth',
          oauth: {
            accessToken: 'oauth-token',
          },
          models: [{ modelId: 'gpt-5' }],
        },
      },
    }

    const model = getModel(sessionSettings, globalSettings, newConfigs(), mockDependencies)

    expect(model).toBeInstanceOf(OpenAIResponses)
  })

  it('keeps the API-key OpenAI model and custom host when desktop OAuth state is read on mobile', () => {
    const sessionSettings: SessionSettings = {
      provider: ModelProviderEnum.OpenAI,
      modelId: 'gpt-4o-mini',
      stream: true,
    }
    const defaultSettings = getDefaultSettings()
    const globalSettings: Settings = {
      ...defaultSettings,
      providers: {
        ...defaultSettings.providers,
        [ModelProviderEnum.OpenAI]: {
          apiKey: 'mobile-api-key',
          apiHost: 'https://mobile-proxy.test/v1',
          activeAuthMode: 'oauth',
          oauth: { accessToken: 'desktop-oauth-token' },
          models: [{ modelId: 'gpt-4o-mini' }],
        },
      },
    }

    const model = getModel(sessionSettings, globalSettings, newConfigs(), {
      ...mockDependencies,
      platformType: 'mobile',
    })

    expect(model).toBeInstanceOf(OpenAI)
    expect((model as OpenAI).options.apiKey).toBe('mobile-api-key')
    expect((model as OpenAI).options.apiHost).toBe('https://mobile-proxy.test/v1')
  })

  it('inherits built-in tool capabilities for an older persisted DeepSeek model', () => {
    const sessionSettings: SessionSettings = {
      provider: ModelProviderEnum.DeepSeek,
      modelId: 'deepseek-v4-flash',
    }
    const defaultSettings = getDefaultSettings()
    const globalSettings: Settings = {
      ...defaultSettings,
      providers: {
        ...defaultSettings.providers,
        [ModelProviderEnum.DeepSeek]: {
          apiKey: 'test-key',
          models: [{ modelId: 'deepseek-v4-flash' }],
        },
      },
    }

    const model = getModel(sessionSettings, globalSettings, newConfigs(), mockDependencies)

    expect(model.isSupportToolUse('agent')).toBe(true)
  })
})
