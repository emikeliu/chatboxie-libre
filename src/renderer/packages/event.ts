import type { AnalyticsEventParams } from '@shared/analytics'

// Tracking has been removed from this edition. This stub keeps existing call
// sites compiling while sending no data anywhere.
export function trackingEvent(_name: string, _params: AnalyticsEventParams = {}) {}
