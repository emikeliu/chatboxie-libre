import { getSystemProviders } from '@shared/providers'
import { ModelProviderEnum } from '@shared/types'
import { describe, expect, it } from 'vitest'
import { FEATURED_PROVIDER_IDS } from './providerIcons'

describe('Provider Spotlight and List integration', () => {
  it('registers exactly the providers this edition ships', () => {
    expect(getSystemProviders().map((provider) => provider.id)).toEqual([
      ModelProviderEnum.LlamaCpp,
      ModelProviderEnum.OpenAI,
      ModelProviderEnum.Gemini,
      ModelProviderEnum.Claude,
      ModelProviderEnum.DeepSeek,
      ModelProviderEnum.Ollama,
    ])
  })

  it.each([
    ModelProviderEnum.OpenAI,
    ModelProviderEnum.Claude,
    ModelProviderEnum.Gemini,
    ModelProviderEnum.DeepSeek,
    ModelProviderEnum.Ollama,
  ])('keeps %s featured so it appears in the Popular group', (id) => {
    expect(getSystemProviders().some((provider) => provider.id === id)).toBe(true)
    expect(FEATURED_PROVIDER_IDS).toContain(id)
  })

  it('keeps llama.cpp reachable even though it is not in the featured list', () => {
    const allProviders = getSystemProviders()
    expect(allProviders.some((provider) => provider.id === ModelProviderEnum.LlamaCpp)).toBe(true)
  })
})
