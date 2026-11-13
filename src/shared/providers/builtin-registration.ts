/**
 * Ordered side-effect entrypoint for the built-in provider registry.
 *
 * Preserve this import order: the registry's Map order is also the default provider
 * display order. Hosts that need model creation import this module explicitly through
 * their model-runtime entrypoint.
 */
// Libre edition: only these providers are registered (display order = this order).
import './definitions/llama-cpp'
import './definitions/openai'
import './definitions/gemini'
import './definitions/claude'
import './definitions/deepseek'
import './definitions/ollama'
