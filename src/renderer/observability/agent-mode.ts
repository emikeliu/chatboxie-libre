/**
 * Error reporting has been removed from this edition. This stub keeps existing
 * call sites compiling while sending nothing anywhere.
 */
export function captureAgentModeException(
  _error: unknown,
  _context: {
    operation:
      | 'suggestion'
      | 'suggestion_model'
      | 'generation'
      | 'tool_pause_continue'
      | 'tool_retry'
      | 'full_access_bypass'
    provider?: string
    model?: string
    agentMode?: string
    fullAccess?: boolean
    toolName?: string
    pauseType?: string
    operationType?: string
  }
) {}
