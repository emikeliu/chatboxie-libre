import { NetworkError } from '../models/errors'
import { getChatboxAPIOrigin } from './chatbox-api-origin'

export { getChatboxAPIOrigin } from './chatbox-api-origin'

const DEFAULT_POOL = [
  'https://api.chatboxai.app',
  'https://api.chatbox.run',
  'https://api.chatboxai.com',
  'https://api.ai-chatbox.com',
  'https://api.chatboxapp.xyz',
]
const PRIMARY_ALIASES = ['https://chatboxai.app']

/**
 * The Chatbox AI cloud service has been removed from this edition. Every request
 * to a Chatbox API origin fails immediately, which keeps account sign-in, license
 * activation/validation, quota lookups, document parsing, web search and the model
 * gateway inert without needing to unwind the features that call them.
 */
export const CHATBOX_CLOUD_SERVICE_ENABLED = false

/** True when the request targets a Chatbox API origin (used to inject auth/platform headers). */
export function isChatboxAPI(input: RequestInfo | URL) {
  const url = typeof input === 'string' ? input : ((input as Request).url ?? input.toString())
  let origin: string
  try {
    origin = new URL(url).origin
  } catch {
    return false
  }
  return DEFAULT_POOL.includes(origin) || PRIMARY_ALIASES.includes(origin) || origin === getChatboxAPIOrigin()
}

/** Every domain the removed Chatbox AI cloud service used to serve, including its API subdomains. */
const CHATBOX_CLOUD_HOST_SUFFIXES = [
  'chatboxai.app',
  'chatbox.run',
  'chatboxai.com',
  'ai-chatbox.com',
  'chatboxapp.xyz',
]

function isChatboxCloudHost(input: RequestInfo | URL): boolean {
  try {
    const url = typeof input === 'string' ? input : input instanceof URL ? input.href : input.url
    const hostname = new URL(url).hostname.toLowerCase()
    return CHATBOX_CLOUD_HOST_SUFFIXES.some((suffix) => hostname === suffix || hostname.endsWith(`.${suffix}`))
  } catch {
    return false
  }
}

function requestOrigin(input: RequestInfo | URL): string {
  try {
    const url = typeof input === 'string' ? input : input instanceof URL ? input.href : input.url
    return new URL(url).origin
  } catch {
    return ''
  }
}

/** Error that blocks a request to the removed Chatbox AI service, or null when the request may proceed. */
export function chatboxCloudServiceBlockError(input: RequestInfo | URL): NetworkError | null {
  if (CHATBOX_CLOUD_SERVICE_ENABLED) return null
  if (!isChatboxCloudHost(input)) return null
  return new NetworkError('Chatbox AI service is not available in this edition.', requestOrigin(input))
}

// Origin probing / fallback has been removed from this edition. These stubs
// keep existing call sites compiling while performing no network activity.
export function reportChatboxAPINetworkError(): void {}

export function testApiOrigins(): Promise<string | null> {
  return Promise.resolve(null)
}

export function startChatboxAPIOriginProbing(_options?: unknown): () => void {
  return () => {}
}

export function resetChatboxAPIOriginProbingForTest(): void {}
