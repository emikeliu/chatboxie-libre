/**
 * Provider-id aliases for icon lookup, shared by the renderer
 * (ProviderImageIcon) and the native ProviderLogo: regional/portal provider
 * variants reuse the parent brand's icon asset.
 */
export const PROVIDER_ICON_ALIASES: Record<string, string> = {
  // llama.cpp uses the llama mark, which ships with the Ollama icon asset.
  'llama.cpp': 'ollama',
  'qwen-portal': 'qwen',
  'minimax-cn': 'minimax',
  'moonshot-cn': 'moonshot',
  'xiaomi-mimo': 'xiaomi',
  'zhipu-glm-coding-plan': 'chatglm-6b',
}
