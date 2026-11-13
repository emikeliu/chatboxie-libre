import type { SentryAdapter, SentryScope } from '../../shared/utils/sentry_adapter'

/**
 * Error reporting has been removed from this edition. This adapter keeps the
 * shared dependency contract intact while sending nothing anywhere.
 */
export class MainSentryAdapter implements SentryAdapter {
  captureException(_error: unknown): void {}

  withScope(callback: (scope: SentryScope) => void): void {
    const scope: SentryScope = {
      setTag(): void {},
      setExtra(): void {},
    }
    callback(scope)
  }
}

export const sentry = new MainSentryAdapter()

export function flushSentry(_timeout: number): Promise<boolean> {
  return Promise.resolve(true)
}
