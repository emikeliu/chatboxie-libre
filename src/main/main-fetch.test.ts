import { describe, expect, it, vi } from 'vitest'

const { electronFetch } = vi.hoisted(() => ({
  electronFetch: vi.fn(),
}))

vi.mock('electron', () => ({
  net: {
    fetch: electronFetch,
  },
}))

import { desktopSessionFetch } from './main-fetch'

describe('desktopSessionFetch', () => {
  it('uses Electron default-session fetch for desktop compatibility requests', async () => {
    const response = new Response('ok')
    electronFetch.mockResolvedValue(response)
    const init: RequestInit = { headers: { authorization: 'Bearer secret' } }

    await expect(desktopSessionFetch(new URL('https://provider.example/v1/models'), init)).resolves.toBe(response)

    expect(electronFetch).toHaveBeenCalledWith('https://provider.example/v1/models', init)
  })
})
