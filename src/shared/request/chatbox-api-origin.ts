const DEFAULT_CHATBOX_API_ORIGIN = 'https://api.chatboxai.app'

type ChatboxAPIOriginEnvironment = Readonly<Record<string, string | undefined>>

export function getChatboxAPIOrigin(environment: ChatboxAPIOriginEnvironment = process.env): string {
  if (environment.USE_LOCAL_API) return 'http://localhost:8002'
  if (environment.USE_BETA_API) return 'https://api-beta.chatboxai.app'
  if (environment.USE_NEWDB_API) return 'https://beta-new-db.chatboxai.app'
  return DEFAULT_CHATBOX_API_ORIGIN
}

// Dynamic origin selection (probing) has been removed from this edition.
export function setDynamicChatboxAPIOrigin(_origin: string): void {}

export function isChatboxAPIOriginOverridden(): boolean {
  return false
}
