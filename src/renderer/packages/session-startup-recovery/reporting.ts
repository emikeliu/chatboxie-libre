import platform from '@/platform'
import type { SessionStartupDiagnostic } from './diagnostics'

export async function getDiagnosticEnvironment() {
  const [version, os] = await Promise.all([
    platform.getVersion().catch(() => 'unknown'),
    platform.getPlatform().catch(() => 'unknown'),
  ])
  const safe = (value: string) => (/^[A-Za-z0-9._+ -]{1,80}$/.test(value) ? value : 'unknown')
  return { version: safe(version), platform: `${platform.type}/${safe(os)}` }
}

/**
 * Remote startup-failure reporting has been removed from this edition. This
 * stub keeps the recovery UI working while sending nothing anywhere.
 */
export function reportStartupFailure(
  _sessionId: string,
  _diagnostic: SessionStartupDiagnostic
): Promise<string | undefined> {
  return Promise.resolve(undefined)
}
