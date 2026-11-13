import { net } from 'electron'
import type { FetchImplementation } from '../shared/request/request'

function getRequestUrl(input: RequestInfo | URL): string {
  if (typeof input === 'string') return input
  if (input instanceof URL) return input.toString()
  return input.url
}

/** Use the default Electron session so Desktop direct requests honor the app's proxy configuration. */
export const desktopSessionFetch: FetchImplementation = (input, init) => {
  return net.fetch(getRequestUrl(input), init)
}
