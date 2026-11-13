export interface ReportErrorContext {
  domain: string
  extras?: Record<string, unknown>
  handled?: boolean
  operation: string
  priority?: string
  tags?: Record<string, string | number | boolean>
}

/**
 * Error reporting has been removed from this edition. Failures stay in local
 * logs and user-facing UI only; nothing is sent to any remote service.
 */
export function reportError(_error: unknown, _context: ReportErrorContext): void {}
