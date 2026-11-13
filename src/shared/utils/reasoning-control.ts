import { pickPersistableProviderMetadata } from '../models/provider-part-metadata'
import { isDeepSeekReasoningEffortModel, isDeepSeekReasoningModel } from '../models/utils/deepseek'
import { usesBedrockConverseProtocol } from '../providers/api-style'
import type { MessageContentParts, ModelProvider, ProviderModelInfo, ProviderOptions } from '../types'
import { ModelProviderEnum } from '../types'
import {
  canDisableGoogleThinking,
  type GoogleThinkingLevel,
  getGoogleThinkingMode,
  getSupportedGoogleThinkingLevels,
} from './google-thinking'

// 'default' sends no reasoning-related parameters at all (the provider's server-side
// default applies); 'off' force-sends the provider's explicit disable parameters — except
// on OpenAI-style routes, where 'off' means "do not send reasoning_effort at all".
export type ReasoningControlLevel =
  | 'default'
  | 'off'
  | 'none'
  | 'minimal'
  | 'low'
  | 'medium'
  | 'high'
  | 'xhigh'
  | 'max'
  | 'extreme'

/**
 * reasoning_effort values forwarded verbatim on OpenAI-family routes. The set is
 * deliberately provider-agnostic: OpenAI-compatible endpoints differ in which values they
 * accept, so any of them may be selected and the endpoint decides. `off` is not part of
 * this list — it is the "send nothing" level.
 */
export const OPENAI_REASONING_EFFORT_LEVELS = [
  'none',
  'minimal',
  'low',
  'medium',
  'high',
  'xhigh',
  'max',
  'extreme',
] as const

export type OpenAIReasoningEffortLevel = (typeof OPENAI_REASONING_EFFORT_LEVELS)[number]

/**
 * Session-level storage for reasoning provider options. `providerOptionsByModel`
 * scopes options to the provider+model they were written for, so switching models
 * within a session never sends parameters a different model cannot accept. The
 * legacy shared `providerOptions` field is never read anymore (pre-map sessions
 * simply start from the 'default' level again); writes clear it.
 *
 * The per-model binding also prepares for the planned UI move: thinking levels
 * will be edited from the model picker (an "edit thinking" affordance per model)
 * instead of the input-box control, and that UI needs each model's level stored
 * independently.
 */
export interface ReasoningProviderOptionsSource {
  providerOptions?: ProviderOptions
  providerOptionsByModel?: Record<string, ProviderOptions>
}

export function getReasoningOptionsModelKey(provider: string | undefined, modelId: string | undefined): string {
  return `${provider || ''}:${modelId || ''}`
}

/**
 * Resolves the reasoning provider options that apply to the given provider+model.
 * Options never cross model boundaries — a model without its own map entry gets
 * `undefined` (the 'default' level), not another model's parameters.
 */
export function resolveReasoningProviderOptions(
  source: ReasoningProviderOptionsSource | undefined,
  provider: string | undefined,
  modelId: string | undefined
): ProviderOptions | undefined {
  return source?.providerOptionsByModel?.[getReasoningOptionsModelKey(provider, modelId)]
}

/**
 * Returns the settings patch that stores `next` for the given provider+model.
 * Writes the per-model map (deleting the entry when `next` is undefined, i.e.
 * the 'default' level) and clears the legacy shared field so the map is the
 * single source of truth from then on.
 */
export function setReasoningProviderOptionsForModel(
  source: ReasoningProviderOptionsSource | undefined,
  provider: string | undefined,
  modelId: string | undefined,
  next: ProviderOptions | undefined
): ReasoningProviderOptionsSource {
  const key = getReasoningOptionsModelKey(provider, modelId)
  const byModel: Record<string, ProviderOptions> = { ...source?.providerOptionsByModel }
  if (next) {
    byModel[key] = next
  } else {
    delete byModel[key]
  }
  return { providerOptions: undefined, providerOptionsByModel: byModel }
}

export type ReasoningControlDisabledReason =
  | 'requires-anthropic-api-style'
  | 'requires-google-api-style'
  | 'requires-openai-api-style'
  | 'requires-deepseek-api-style'
  | 'requires-qwen-api-style'
  | 'requires-xai-api-style'

export interface ReasoningControlCapabilities {
  supported: boolean
  kind:
    | 'anthropic-adaptive-effort'
    | 'anthropic-effort'
    | 'budget'
    | 'deepseek-effort'
    | 'level'
    | 'openai-effort'
    | 'openrouter-reasoning'
    | 'toggle'
    | 'xai-effort'
  disabledReason?: ReasoningControlDisabledReason
}

export interface ReasoningControlOption {
  level: ReasoningControlLevel
  label: ReasoningControlLevel | 'on'
}

const DEFAULT_CAPABILITIES: ReasoningControlCapabilities = {
  supported: false,
  kind: 'toggle',
}

// Levels a provider-specific control can express (terse UI levels). OpenAI routes can also
// carry the extended effort vocabulary, which is coerced down for these providers.
type ReasoningEffortLevel = 'low' | 'medium' | 'high'
type DeepSeekReasoningEffort = 'low' | 'high' | 'max'

/**
 * Maps an extended OpenAI effort level onto a provider-specific terse level: values below
 * low collapse to low, values above high collapse to high. Levels providers never offer
 * (because their option list is terse) are only reachable from stale data.
 */
function toReasoningEffortLevel(level: ReasoningControlLevel): ReasoningEffortLevel {
  switch (level) {
    case 'none':
    case 'minimal':
    case 'low':
      return 'low'
    case 'medium':
      return 'medium'
    default:
      return 'high'
  }
}

const CLAUDE_BUDGET_BY_LEVEL: Record<ReasoningEffortLevel, number> = {
  low: 1024,
  medium: 4096,
  high: 8192,
}

const GEMINI_BUDGET_BY_LEVEL: Record<ReasoningEffortLevel, number> = {
  low: 1024,
  medium: 8192,
  high: 24576,
}

// Readback boundaries accept both the level budgets above (1024/8192/24576) and the
// presets written by the legacy session-settings modal (2048/5120/10240), so upgraded
// sessions keep displaying the level the user originally chose.
const GEMINI_LEVEL_READBACK_MIN: Record<Exclude<ReasoningEffortLevel, 'low'>, number> = {
  medium: 4096,
  high: 10240,
}

// DeepSeek's official APIs expose low/high/max rather than the UI's
// low/medium/high. Preserve three distinct controls by mapping the middle UI
// level to DeepSeek high and the highest UI level to DeepSeek max.
const DEEPSEEK_EFFORT_BY_LEVEL: Record<ReasoningEffortLevel, DeepSeekReasoningEffort> = {
  low: 'low',
  medium: 'high',
  high: 'max',
}

const QWEN_THINKING_BUDGET_BY_LEVEL: Record<ReasoningEffortLevel, number> = {
  low: 1024,
  medium: 4096,
  high: 8192,
}

const GPT_VERSION_PATTERN = /(?:^|\/)gpt-(\d+)(?=[.-]|$)/i
const GPT_EFFORT_MODELS = [/(?:^|\/)gpt-oss(?:[.-]|$)/i, /(?:^|\/)o[1-9](?:[.-]|$)/i]
// o-series models do not accept minimal/none to disable thinking.
const OPENAI_NO_DISABLE_MODELS = [/(?:^|\/)o[1-9](?:[.-]|$)/i]
// Chat-tuned GPT variants are excluded from the generation-based effort fallback.
const GPT_NON_REASONING_CHAT_MODELS = [/(?:^|\/)gpt-\d+[\w.-]*[.-]chat(?:[.-]|$)/i]
// o1-preview and o1-mini predate the reasoning_effort parameter — the API rejects it
// for them entirely, so they must not get effort controls at all.
const OPENAI_NO_EFFORT_PARAM_MODELS = [/(?:^|\/)o1-(?:preview|mini)(?:[.-]|$)/i]
const CLAUDE_EFFORT_MODELS = [/(?:^|\/)claude-opus-4-5(?:[.-]|$)/i]
const CLAUDE_ADAPTIVE_EFFORT_MODELS = [/(?:^|\/)claude-opus-4-(?:7|8)(?:[.-]|$)/i, /(?:^|\/)claude-opus-5(?:[.-]|$)/i]
const CLAUDE_BUDGET_MODELS = [
  /(?:^|\/)claude-3-7-sonnet/i,
  /(?:^|\/)claude-sonnet-4/i,
  /(?:^|\/)claude-haiku-4-5/i,
  /(?:^|\/)claude-opus-4(?![.-]?5)(?![.-]?7)(?![.-]?8)/i,
]
const QWEN_THINKING_MODELS = [/^qwen3/i, /(?:^|\/)qwen3/i]
const GROK_REASONING_EFFORT_MODELS = [
  /(?:^|\/)grok-4\.3(?:-latest)?$/i,
  /(?:^|\/)grok-4(?:-latest|-0709)?$/i,
  /(?:^|\/)grok-4-fast(?:-reasoning)?(?:-latest)?$/i,
  /(?:^|\/)grok-4-1-fast(?:-reasoning)?(?:-latest)?$/i,
]

function matchesAny(modelId: string, patterns: RegExp[]): boolean {
  return patterns.some((pattern) => pattern.test(modelId))
}

// Models whose thinking cannot be force-disabled do not get an 'off' option; selecting
// 'off' for them falls back to 'default' (send nothing). Claude effort/adaptive models
// are only controlled via the effort request param — the AI SDK never emits
// `thinking: {type: 'disabled'}`, so an explicit off cannot be expressed on the wire.
// Gemini 2.5 Pro enforces a minimum thinking budget, so thinkingBudget: 0 is rejected.
function supportsExplicitDisable(
  kind: ReasoningControlCapabilities['kind'],
  effectiveProvider: ModelProvider | undefined,
  modelId: string
): boolean {
  if (effectiveProvider === ModelProviderEnum.Gemini && !canDisableGoogleThinking(modelId)) {
    return false
  }
  if (kind === 'anthropic-adaptive-effort' || kind === 'anthropic-effort') {
    return false
  }
  // OpenAI-style routes always offer 'off', which there means "send no reasoning_effort at
  // all" rather than an explicit disable parameter.
  if (kind === 'openai-effort') {
    return true
  }
  if (isOpenAIStyleEffectiveProvider(effectiveProvider) && !supportsOpenAIThinkingDisable(modelId)) {
    return false
  }
  return true
}

function getGptMajorVersion(modelId: string): number {
  return Number(GPT_VERSION_PATTERN.exec(modelId)?.[1] ?? 0)
}

function supportsOpenAIThinkingDisable(modelId: string): boolean {
  // New GPT generations inherit effort controls, but an accepted off value must be
  // established separately before offering minimal/none in the UI or request options.
  return getGptMajorVersion(modelId) <= 5 && !matchesAny(modelId, OPENAI_NO_DISABLE_MODELS)
}

function isGptEffortModel(modelId: string): boolean {
  return (
    (getGptMajorVersion(modelId) >= 5 || matchesAny(modelId, GPT_EFFORT_MODELS)) &&
    !matchesAny(modelId, GPT_NON_REASONING_CHAT_MODELS) &&
    !matchesAny(modelId, OPENAI_NO_EFFORT_PARAM_MODELS)
  )
}

/**
 * Claude models that use adaptive thinking effort instead of a token budget.
 * Shared with the Claude provider so capability detection and request
 * construction stay in sync.
 */
export function isClaudeAdaptiveThinkingModel(modelId: string): boolean {
  return matchesAny(modelId, CLAUDE_ADAPTIVE_EFFORT_MODELS)
}

/**
 * Claude models whose thinking is controlled by the effort request param
 * (Opus 4.5 and the adaptive 4.7/4.8/5 generations) rather than a token budget.
 */
export function usesClaudeEffortControl(modelId: string): boolean {
  return matchesAny(modelId, [...CLAUDE_EFFORT_MODELS, ...CLAUDE_ADAPTIVE_EFFORT_MODELS])
}

/**
 * Drops Claude reasoning options written for a different model generation so they are
 * never sent on the wire: effort/adaptive models only accept the effort param, while
 * budget-style models only accept the thinking config. Without this, options persisted
 * on a session leak through model switches (e.g. a Sonnet thinking budget sent to an
 * adaptive Opus model) and contradict the 'default' (send nothing) level readback.
 */
export function normalizeClaudeReasoningOptions(
  modelId: string,
  claude: ProviderOptions['claude']
): ProviderOptions['claude'] {
  if (!claude) return undefined
  if (usesClaudeEffortControl(modelId)) {
    if (!claude.effort) return undefined
    // 'xhigh'/'max' are DeepSeek-Anthropic-only values; clamp to the strongest
    // level Claude effort models accept.
    return { effort: claude.effort === 'xhigh' || claude.effort === 'max' ? 'high' : claude.effort }
  }
  // Budget-style models reject enabled thinking without budget_tokens (a shape only
  // the DeepSeek-Anthropic writer produces); treat it as 'default' instead.
  if (claude.thinking?.type === 'enabled' && claude.thinking.budgetTokens == null) return undefined
  return claude.thinking ? { thinking: claude.thinking } : undefined
}

/**
 * Whether a reasoning_effort value can be sent to the given OpenAI-style model.
 *
 * OpenAI-compatible endpoints differ in which effort values they accept, so the client no
 * longer filters by model id — whatever the user selects is forwarded and the endpoint
 * decides (an unsupported value surfaces as a normal API error). Only o1-preview/o1-mini
 * are excluded: they predate the parameter entirely and reject the request.
 */
export function isOpenAIReasoningEffortSupported(modelId: string, _effort: string): boolean {
  return !matchesAny(modelId, OPENAI_NO_EFFORT_PARAM_MODELS)
}

/**
 * Drops OpenAI reasoning options the target model cannot accept (see
 * isOpenAIReasoningEffortSupported) — a stale off effort persisted under a GPT-5
 * session must not survive a switch to an o-series model. Returns undefined when the
 * options are invalid for the model, dropping the whole (reasoning-only) namespace.
 */
export function normalizeOpenAIReasoningOptions(
  modelId: string,
  openai: ProviderOptions['openai']
): ProviderOptions['openai'] {
  if (!openai) return undefined
  if (matchesAny(modelId, OPENAI_NO_EFFORT_PARAM_MODELS)) {
    return undefined
  }
  if (openai.reasoningEffort !== undefined && !isOpenAIReasoningEffortSupported(modelId, openai.reasoningEffort)) {
    return undefined
  }
  return openai
}

/**
 * Select the OpenAI reasoning options that have a defined OpenAI-compatible wire mapping.
 * `@ai-sdk/openai-compatible` forwards unknown keys verbatim, so OpenAI-only flags such as
 * forceReasoning, reasoningSummary and include must never cross this boundary.
 *
 * Keep this whitelist in sync with the OpenAI option writers in getReasoningProviderOptions.
 * Add a key here only after its compatible request-body mapping is verified.
 */
export function pickOpenAICompatibleReasoningOptions(
  modelId: string,
  providerOptions: ProviderOptions | undefined
): ProviderOptions['openaiCompatible'] {
  const reasoningEffort = providerOptions?.openai?.reasoningEffort ?? providerOptions?.openaiCompatible?.reasoningEffort
  if (reasoningEffort === undefined) return undefined
  if (!isOpenAIReasoningEffortSupported(modelId, reasoningEffort)) return undefined
  return { reasoningEffort }
}

function isOpenAIStyleEffectiveProvider(provider: ModelProvider | undefined): boolean {
  return (
    provider === ModelProviderEnum.OpenAI ||
    provider === ModelProviderEnum.OpenAIResponses ||
    provider === ModelProviderEnum.Azure
  )
}

/**
 * Routes whose request body carries the OpenAI wire format, where thinking is controlled
 * with `reasoning_effort` (Chat Completions) or `reasoning.effort` (Responses). Covers
 * built-in OpenAI-style providers, the ChatboxAI/OpenCode gateways, and user-created
 * providers whose models are served over an OpenAI-compatible endpoint.
 */
function isOpenAIWireRoute(provider: ModelProvider | undefined, model: ProviderModelInfo): boolean {
  // o1-preview/o1-mini reject the reasoning_effort parameter entirely.
  if (matchesAny(model.modelId, OPENAI_NO_EFFORT_PARAM_MODELS)) return false
  if (isOpenAICompatibleApiStyle(provider, model)) return true
  return isOpenAIStyleEffectiveProvider(getEffectiveProvider(provider, model))
}

/**
 * Thinking-effort control for any model served over an OpenAI-compatible endpoint.
 *
 * The endpoint accepts `reasoning_effort` for whatever backend it hosts (Qwen, Claude,
 * DeepSeek, a local llama.cpp, …), so the control is offered regardless of the model id or
 * the model's registry capabilities — refusing by model id is what produced the old
 * "disabled because this Qwen model is not exposed through the Qwen API style" lock.
 * `off` sends no parameter, so an endpoint that ignores the field is unaffected.
 */
function supportsGenericOpenAIEffort(
  provider: ModelProvider | undefined,
  model: ProviderModelInfo | undefined
): boolean {
  if (!model?.modelId) return false
  if (model.type && model.type !== 'chat') return false
  // Every chat model on an OpenAI-compatible endpoint gets the level menu; `off` (the default)
  // sends no parameter, so endpoints/models without reasoning support are unaffected.
  return isOpenAIWireRoute(provider, model)
}

type LegacyOpenAICompatibleReasoning = NonNullable<ProviderOptions['openaiCompatible']>['reasoning']

/**
 * Interprets the legacy `openaiCompatible.reasoning` options (written by older
 * versions) as a DeepSeek-style thinking toggle. Shared with the ChatboxAI
 * gateway model so both read paths agree.
 */
export function getLegacyOpenAICompatibleThinkingType(
  reasoning: LegacyOpenAICompatibleReasoning
): 'enabled' | 'disabled' | undefined {
  if (!reasoning) return undefined
  if (reasoning.enabled === false || reasoning.exclude === true) return 'disabled'
  return reasoning.enabled ? 'enabled' : undefined
}

function getEffectiveProvider(
  provider: ModelProvider | undefined,
  model?: ProviderModelInfo | null
): ModelProvider | undefined {
  if (!model?.apiStyle || !usesModelApiStyleForReasoning(provider)) {
    return provider
  }

  if (provider === ModelProviderEnum.OpenRouter) {
    return provider
  }

  if (model.apiStyle === 'anthropic') return ModelProviderEnum.Claude
  if (model.apiStyle === 'google') return ModelProviderEnum.Gemini
  if (model.apiStyle === 'openai-responses') return ModelProviderEnum.OpenAIResponses
  return ModelProviderEnum.OpenAI
}

// All built-in provider ids. Any id outside this set is a user-created custom provider,
// whose reasoning support must be judged by its API style (provider type) + model id.
const BUILTIN_PROVIDER_IDS = new Set<string>(Object.values(ModelProviderEnum))

function isCustomProviderId(provider: ModelProvider | undefined): boolean {
  return !!provider && !BUILTIN_PROVIDER_IDS.has(provider)
}

function isOpenCodeGatewayProvider(provider: ModelProvider | undefined): boolean {
  return provider === ModelProviderEnum.OpenCodeGo || provider === ModelProviderEnum.OpenCodeZen
}

function usesModelApiStyleForReasoning(provider: ModelProvider | undefined): boolean {
  // ChatboxAI and the OpenCode gateways proxy many backend models, and custom providers
  // wrap an upstream API, so we resolve the effective provider from the model's API style
  // rather than the provider id itself.
  return (
    provider === ModelProviderEnum.ChatboxAI ||
    provider === ModelProviderEnum.Custom ||
    isOpenCodeGatewayProvider(provider) ||
    isCustomProviderId(provider)
  )
}

function isOpenAICompatibleApiStyle(provider: ModelProvider | undefined, model: ProviderModelInfo): boolean {
  // ChatboxAI, the OpenCode gateways, and custom providers can all expose OpenAI-compatible
  // endpoints; treat a missing/`openai` API style as OpenAI-compatible so model-id
  // based detection (e.g. DeepSeek) works the same way for all three.
  return (
    (provider === ModelProviderEnum.ChatboxAI || isOpenCodeGatewayProvider(provider) || isCustomProviderId(provider)) &&
    (!model.apiStyle || model.apiStyle === 'openai')
  )
}

function usesOpenAICompatibleReasoningHistory(provider: ModelProvider | undefined, model: ProviderModelInfo): boolean {
  if (model.apiStyle === 'openai') return true

  // getModel() stamps the provider type onto current models. Keep the missing-style
  // fallback for older ChatboxAI/OpenCode/custom model records that predate that metadata.
  return (
    !model.apiStyle &&
    (provider === ModelProviderEnum.ChatboxAI ||
      provider === ModelProviderEnum.Custom ||
      isOpenCodeGatewayProvider(provider) ||
      isCustomProviderId(provider))
  )
}

/**
 * Whether historical assistant reasoning must survive conversion to model messages.
 *
 * The native DeepSeek SDK needs reasoning parts for its V4 request conversion.
 * OpenAI-compatible routes, including built-in and custom providers, need the same
 * parts so their provider SDK can emit reasoning history. Keeping both the API-style
 * and model-family checks avoids reintroducing the Grok, Mistral, and Gemini regressions
 * caused by preserving reasoning for every provider.
 */
export function shouldPreserveDeepSeekReasoning(
  provider: ModelProvider | undefined,
  model: ProviderModelInfo | null | undefined
): boolean {
  if (provider === ModelProviderEnum.DeepSeek) return true
  return !!model && usesOpenAICompatibleReasoningHistory(provider, model) && isDeepSeekReasoningModel(model.modelId)
}

export interface ReasoningReplayPolicy {
  /** Which assistant reasoning history survives conversion to model messages. */
  preserveReasoning: false | 'all-turns'
  /** Whether only blocks carrying whitelisted replay metadata go on the wire. */
  signedReasoningOnly: boolean
  /**
   * Provider namespaces whose replay metadata may go on the wire for this
   * route. Replay is per-route, not global: a session can switch providers
   * between turns, so a stored block can carry another provider's metadata.
   * Sending that foreign metadata would turn an otherwise-dropped unsigned
   * block into a wire-visible one (Anthropic rejects thinking blocks without a
   * valid signature; OpenAI Responses warns on reasoning parts it cannot
   * reconstruct). Applies to text parts as well as reasoning ones, because
   * Gemini's signature rides on the last text part of a response with no
   * function call. Empty for the plain-text DeepSeek channel, which replays
   * text only.
   */
  replayNamespaces: readonly string[]
  /** Replay source of the target model; Responses ciphertext replays only to its own source. */
  reasoningReplaySource?: string
}

/**
 * How assistant reasoning history is replayed to the target route.
 *
 * Anthropic Messages routes (direct Claude, ChatboxAI/custom providers with
 * `apiStyle === 'anthropic'`) replay signed thinking on every assistant turn —
 * the documented pattern ("pass everything back; the API automatically
 * filters"), and signatures are valid across Claude API / Bedrock / Vertex.
 * Unsigned / foreign reasoning (Kimi, DeepSeek, Gemini thoughts, data saved
 * before signature capture) is omitted at convert time via the signature
 * whitelist, not by wiping storage. Bedrock Converse shares the apiStyle but
 * expects the `bedrock` namespace and empty text blocks — it stays excluded
 * until that protocol is implemented.
 *
 * OpenAI Responses is the other signed-replay route: Chatbox forces
 * `store: false` and requests `reasoning.encrypted_content`, so replaying the
 * encrypted reasoning item is the only way to keep the model's reasoning across
 * turns (the API has no server-side state to fall back on). Only blocks that
 * actually carry an item id / encrypted payload go out — a plain-text thought
 * saved on another route cannot be reconstructed into an OpenAI reasoning item.
 *
 * Gemini is the third signed-replay route. Its `thoughtSignature` is mandatory on
 * `functionCall` parts for Gemini 3 (the request 400s without it) and is also
 * attached to the last `text` / `thought` part of a response with no function
 * call, where Google documents that omitting it can degrade performance. Only
 * blocks that actually carry a signature go out — unsigned thoughts from other
 * providers are dropped rather than replayed as Gemini thought parts.
 *
 * DeepSeek thinking mode keeps its existing all-turns plain-text behavior (see
 * `shouldPreserveDeepSeekReasoning`).
 */
export function resolveReasoningReplayPolicy(
  provider: ModelProvider | undefined,
  model: ProviderModelInfo | null | undefined,
  reasoningReplaySource?: string
): ReasoningReplayPolicy {
  const policy = resolveReasoningReplayRoute(provider, model)
  return reasoningReplaySource ? { ...policy, reasoningReplaySource } : policy
}

function resolveReasoningReplayRoute(
  provider: ModelProvider | undefined,
  model: ProviderModelInfo | null | undefined
): ReasoningReplayPolicy {
  if (model?.apiStyle === 'anthropic' && !usesBedrockConverseProtocol(provider)) {
    return { preserveReasoning: 'all-turns', signedReasoningOnly: true, replayNamespaces: ['anthropic'] }
  }
  if (model?.apiStyle === 'openai-responses') {
    return { preserveReasoning: 'all-turns', signedReasoningOnly: true, replayNamespaces: ['openai'] }
  }
  if (model?.apiStyle === 'google') {
    return { preserveReasoning: 'all-turns', signedReasoningOnly: true, replayNamespaces: ['google'] }
  }
  if (shouldPreserveDeepSeekReasoning(provider, model)) {
    return { preserveReasoning: 'all-turns', signedReasoningOnly: false, replayNamespaces: [] }
  }
  return { preserveReasoning: false, signedReasoningOnly: false, replayNamespaces: [] }
}

/**
 * Budget-style Claude can send `thinking: { type: 'disabled' }`. Effort /
 * adaptive models cannot express an explicit off on the wire.
 */
export function canSendClaudeThinkingDisabled(modelId: string | undefined): boolean {
  return !!modelId && !usesClaudeEffortControl(modelId)
}

/**
 * Anthropic manual (budget) thinking enforces that the final assistant *turn*
 * of a thinking-enabled request begins with a thinking block. A tool-use loop
 * is one turn, and in non-interleaved mode only the loop's first response
 * carries thinking — later steps legitimately ship bare `tool_use` blocks. The
 * check therefore anchors on the turn start, not on the last tool call:
 * resuming is safe whenever the first block the converter puts on the wire is
 * a signed thinking block, even if later steps in the same turn are unsigned.
 *
 * When the resumed turn cannot open with signed thinking (data saved before
 * signature capture, or a turn minted by another provider), budget-style
 * models turn thinking off for this request instead — the official
 * graceful-degradation path ("To avoid this requirement, disable thinking").
 * The caller must then also drop reasoning replay so the degraded request
 * carries no thinking blocks at all.
 *
 * Effort/adaptive Claude models are exempt by rule, not as a fallback for
 * their missing `thinking: {type: 'disabled'}` wire shape: the turn-start
 * requirement exists only in manual mode ("Adaptive mode relaxes this: no
 * assistant turn needs to start with one"), and these models never receive
 * manual thinking config anyway (normalizeClaudeReasoningOptions keeps only
 * the effort param). An unsigned resume is already a valid request for them —
 * there is nothing to degrade.
 *
 * Anchored on the last prompt message: only a request that ends with an
 * assistant tool exchange (a resumed continuation) is constrained. A fresh
 * user turn ends with the user message, its history needs no thinking blocks
 * ("outside tool use, omit prior turns' thinking"), and must never have its
 * thinking silently degraded by old unsigned turns earlier in the session.
 */
export function shouldDisableClaudeThinkingForUnsignedResume(
  lastPromptMessage: { role: string; contentParts?: MessageContentParts } | undefined,
  /**
   * Whether this request replays Anthropic signed thinking. Callers must pass
   * the route's Anthropic-ness, not merely "signed replay is on": OpenAI
   * Responses also replays signed metadata but has no turn-start rule, and
   * degrading its thinking would silently drop the encrypted reasoning items
   * this request exists to send.
   */
  anthropicSignedReplay: boolean,
  modelId: string
): boolean {
  if (!anthropicSignedReplay || !canSendClaudeThinkingDisabled(modelId)) return false
  if (lastPromptMessage?.role !== 'assistant') return false
  const parts = lastPromptMessage.contentParts ?? []
  // Only completed tool calls reach the wire; without one the request does not
  // end in a resumed tool exchange, so the turn-start rule has nothing to bite.
  if (!parts.some(isWireVisibleToolCall)) return false
  return !turnStartsWithSignedThinking(parts)
}

function isWireVisibleToolCall(part: MessageContentParts[number]): boolean {
  return part.type === 'tool-call' && part.state !== 'call' && part.state !== 'paused'
}

/**
 * Mirrors what `convertToModelMessages` emits under signed-only replay: skips
 * parts that never reach the wire (unsigned reasoning, empty or protocol-only
 * text, unfinished tool calls, info parts) and reports whether the first
 * surviving block is signed thinking.
 *
 * Filtered to the `anthropic` namespace for the same reason the converter is:
 * this only ever runs on Anthropic routes, where a leftover OpenAI reasoning
 * item is dropped rather than emitted, so it must not count as signed thinking.
 */
function turnStartsWithSignedThinking(parts: MessageContentParts): boolean {
  for (const part of parts) {
    switch (part.type) {
      case 'reasoning':
        if (pickPersistableProviderMetadata(part.providerMetadata, ['anthropic'])) return true
        break // unsigned reasoning is omitted from the wire; keep looking
      case 'text':
        if (part.text && !part.protocolOnly) return false
        break
      case 'tool-call':
        if (isWireVisibleToolCall(part)) return false
        break
      case 'image':
        return false
      default:
        break // info/status parts never reach the wire
    }
  }
  return false
}

export function getReasoningControlCapabilities(
  provider: ModelProvider | undefined,
  model?: ProviderModelInfo | null
): ReasoningControlCapabilities {
  const modelId = model?.modelId
  if (!provider || !modelId) {
    return DEFAULT_CAPABILITIES
  }

  const effectiveProvider = getEffectiveProvider(provider, model)
  const isChatboxDeepSeek = isChatboxAIDeepSeekWithOfficialApiStyle(provider, model)
  const disabledReason = getApiStyleDisabledReason(provider, effectiveProvider, model)
  if (disabledReason) {
    return { supported: false, kind: 'toggle', disabledReason }
  }

  // ChatboxAI's server-selected API style may change independently of the client.
  // DeepSeek V4 officially supports effort controls in OpenAI Chat, Anthropic,
  // and Responses formats; older reasoning models retain toggle controls.
  if (isChatboxDeepSeek) {
    return { supported: true, kind: isDeepSeekReasoningEffortModel(modelId) ? 'deepseek-effort' : 'toggle' }
  }
  if (effectiveProvider === ModelProviderEnum.Claude && matchesAny(modelId, CLAUDE_ADAPTIVE_EFFORT_MODELS)) {
    return { supported: true, kind: 'anthropic-adaptive-effort' }
  }
  if (effectiveProvider === ModelProviderEnum.Claude && matchesAny(modelId, CLAUDE_EFFORT_MODELS)) {
    return { supported: true, kind: 'anthropic-effort' }
  }
  if (effectiveProvider === ModelProviderEnum.Claude && matchesAny(modelId, CLAUDE_BUDGET_MODELS)) {
    return { supported: true, kind: 'budget' }
  }
  if (effectiveProvider === ModelProviderEnum.Gemini) {
    const mode = getGoogleThinkingMode(modelId)
    if (mode === 'budget') return { supported: true, kind: 'budget' }
    if (mode === 'level') return { supported: true, kind: 'level' }
  }
  if (effectiveProvider === ModelProviderEnum.DeepSeek && isDeepSeekThinkingModel(model)) {
    return { supported: true, kind: isDeepSeekReasoningEffortModel(modelId) ? 'deepseek-effort' : 'toggle' }
  }
  if (model && isOpenAICompatibleApiStyle(provider, model) && isDeepSeekThinkingModel(model)) {
    return { supported: true, kind: isDeepSeekReasoningEffortModel(modelId) ? 'deepseek-effort' : 'toggle' }
  }
  if (isOpenAIStyleEffectiveProvider(effectiveProvider) && isGptEffortModel(modelId)) {
    return { supported: true, kind: 'openai-effort' }
  }
  if (
    (effectiveProvider === ModelProviderEnum.Qwen || effectiveProvider === ModelProviderEnum.QwenPortal) &&
    matchesAny(modelId, QWEN_THINKING_MODELS)
  ) {
    return { supported: true, kind: 'budget' }
  }
  if (effectiveProvider === ModelProviderEnum.XAI && matchesAny(modelId, GROK_REASONING_EFFORT_MODELS)) {
    return { supported: true, kind: 'xai-effort' }
  }
  if (effectiveProvider === ModelProviderEnum.OpenRouter && isOpenRouterReasoningModel(model)) {
    return { supported: true, kind: 'openrouter-reasoning' }
  }
  // Any model declared as reasoning-capable gets GPT-6-style effort levels when the route
  // speaks the OpenAI wire format. Specific families above win first, so known models keep
  // their provider-correct controls.
  if (supportsGenericOpenAIEffort(provider, model)) {
    return { supported: true, kind: 'openai-effort' }
  }

  return DEFAULT_CAPABILITIES
}

function getApiStyleDisabledReason(
  provider: ModelProvider | undefined,
  effectiveProvider: ModelProvider | undefined,
  model: ProviderModelInfo
): ReasoningControlDisabledReason | undefined {
  if (effectiveProvider === ModelProviderEnum.OpenRouter) {
    return undefined
  }

  // OpenAI wire-compatible routes can always express effort via `reasoning_effort`, even when
  // the model id belongs to another vendor family (qwen3, claude, gemini, grok, …). Returning
  // no reason here lets the generic OpenAI effort levels take over instead of showing a lock.
  if (isOpenAIWireRoute(provider, model)) {
    return undefined
  }

  const modelId = model.modelId
  if (matchesAny(modelId, [...CLAUDE_ADAPTIVE_EFFORT_MODELS, ...CLAUDE_EFFORT_MODELS, ...CLAUDE_BUDGET_MODELS])) {
    if (effectiveProvider !== ModelProviderEnum.Claude) {
      return 'requires-anthropic-api-style'
    }
  }

  if (getGoogleThinkingMode(modelId) !== 'none') {
    if (effectiveProvider !== ModelProviderEnum.Gemini) {
      return 'requires-google-api-style'
    }
  }

  if (isGptEffortModel(modelId) && !isOpenAIStyleEffectiveProvider(effectiveProvider)) {
    return 'requires-openai-api-style'
  }

  if (
    isDeepSeekReasoningModel(modelId) &&
    effectiveProvider !== ModelProviderEnum.DeepSeek &&
    !isOpenAICompatibleApiStyle(provider, model) &&
    !isChatboxAIDeepSeekWithOfficialApiStyle(provider, model)
  ) {
    return 'requires-deepseek-api-style'
  }

  if (
    matchesAny(modelId, QWEN_THINKING_MODELS) &&
    effectiveProvider !== ModelProviderEnum.Qwen &&
    effectiveProvider !== ModelProviderEnum.QwenPortal &&
    !isOpenCodeGatewayProvider(provider)
  ) {
    return 'requires-qwen-api-style'
  }

  if (
    matchesAny(modelId, GROK_REASONING_EFFORT_MODELS) &&
    effectiveProvider !== ModelProviderEnum.XAI &&
    !isOpenCodeGatewayProvider(provider)
  ) {
    return 'requires-xai-api-style'
  }

  return undefined
}

export function getReasoningControlLevel(
  provider: ModelProvider | undefined,
  model: ProviderModelInfo | null | undefined,
  providerOptions?: ProviderOptions
): ReasoningControlLevel {
  const level = deriveReasoningControlLevel(provider, model, providerOptions)
  if (level !== 'off') return level
  if (getReasoningControlCapabilities(provider, model).kind === 'openai-effort') return level
  // Stale options (written by older versions or under another model) can read back as
  // 'off' on a model that no longer offers an off option; report them as 'default' so
  // the displayed level always exists in getReasoningControlOptions.
  const capabilities = getReasoningControlCapabilities(provider, model)
  const effectiveProvider = getEffectiveProvider(provider, model)
  return supportsExplicitDisable(capabilities.kind, effectiveProvider, model?.modelId || '') ? 'off' : 'default'
}

function deriveReasoningControlLevel(
  provider: ModelProvider | undefined,
  model: ProviderModelInfo | null | undefined,
  providerOptions?: ProviderOptions
): ReasoningControlLevel {
  const capabilities = getReasoningControlCapabilities(provider, model)
  if (!capabilities.supported) return 'default'

  const effectiveProvider = getEffectiveProvider(provider, model)
  if (isChatboxAIDeepSeekWithOfficialApiStyle(provider, model)) {
    const modelId = model?.modelId || ''
    if (model?.apiStyle === 'anthropic') {
      return deriveDeepSeekLevel(modelId, providerOptions?.claude?.thinking?.type, providerOptions?.claude?.effort)
    }
    if (model?.apiStyle === 'openai-responses') {
      return deriveDeepSeekResponsesLevel(modelId, providerOptions?.openai?.reasoningEffort)
    }
    return deriveDeepSeekLevel(
      modelId,
      providerOptions?.deepseek?.thinking?.type ??
        getLegacyOpenAICompatibleThinkingType(providerOptions?.openaiCompatible?.reasoning),
      providerOptions?.deepseek?.reasoningEffort
    )
  }
  if (model && isOpenAICompatibleApiStyle(provider, model) && isDeepSeekThinkingModel(model)) {
    return deriveDeepSeekLevel(
      model.modelId,
      providerOptions?.deepseek?.thinking?.type ??
        getLegacyOpenAICompatibleThinkingType(providerOptions?.openaiCompatible?.reasoning),
      providerOptions?.deepseek?.reasoningEffort
    )
  }
  // OpenAI-style effort models read back any of the extended effort values; no stored
  // effort means 'off' (nothing is sent), matching the option list. Checked before the
  // provider branches because OpenAI-style providers are routed by their effective
  // provider id, which would otherwise collapse none/minimal to 'off'.
  if (capabilities.kind === 'openai-effort') {
    return normalizeOpenAIEffortToLevel(
      providerOptions?.openai?.reasoningEffort ?? providerOptions?.openaiCompatible?.reasoningEffort
    )
  }
  if (effectiveProvider === ModelProviderEnum.Claude) {
    if (capabilities.kind === 'anthropic-adaptive-effort' || capabilities.kind === 'anthropic-effort') {
      return normalizeEffortToLevel(providerOptions?.claude?.effort)
    }
    const thinking = providerOptions?.claude?.thinking
    if (!thinking) return 'default'
    if (thinking.type !== 'enabled') return 'off'
    const budget = thinking.budgetTokens ?? 0
    if (budget >= CLAUDE_BUDGET_BY_LEVEL.high) return 'high'
    if (budget >= CLAUDE_BUDGET_BY_LEVEL.medium) return 'medium'
    return 'low'
  }
  if (isOpenAIStyleEffectiveProvider(effectiveProvider)) {
    const effort = providerOptions?.openai?.reasoningEffort
    return normalizeEffortToLevel(effort)
  }
  if (effectiveProvider === ModelProviderEnum.XAI) {
    const effort = providerOptions?.openai?.reasoningEffort
    return normalizeEffortToLevel(effort)
  }
  if (effectiveProvider === ModelProviderEnum.OpenRouter) {
    const reasoning = providerOptions?.openrouter?.reasoning
    if (!reasoning) return 'default'
    if (reasoning.enabled === false) return 'off'
    return normalizeEffortToLevel(reasoning.effort)
  }
  if (effectiveProvider === ModelProviderEnum.Gemini) {
    const config = providerOptions?.google?.thinkingConfig
    if (!config) return 'default'
    if (config.includeThoughts === false) return 'off'
    if (config.thinkingLevel && config.thinkingLevel !== 'minimal') return config.thinkingLevel
    const budget = config.thinkingBudget
    if (budget === undefined || budget <= 0) return 'off'
    if (budget >= GEMINI_LEVEL_READBACK_MIN.high) return 'high'
    if (budget >= GEMINI_LEVEL_READBACK_MIN.medium) return 'medium'
    return 'low'
  }
  if (effectiveProvider === ModelProviderEnum.DeepSeek) {
    return deriveDeepSeekLevel(
      model?.modelId || '',
      providerOptions?.deepseek?.thinking?.type,
      providerOptions?.deepseek?.reasoningEffort
    )
  }
  if (effectiveProvider === ModelProviderEnum.Qwen || effectiveProvider === ModelProviderEnum.QwenPortal) {
    const openaiCompatible = providerOptions?.openaiCompatible
    if (openaiCompatible?.enable_thinking === false) return 'off'
    if (openaiCompatible?.enable_thinking !== true) return 'default'
    const budget = openaiCompatible.thinking_budget
    if (budget !== undefined && budget >= QWEN_THINKING_BUDGET_BY_LEVEL.high) return 'high'
    if (budget !== undefined && budget >= QWEN_THINKING_BUDGET_BY_LEVEL.medium) return 'medium'
    return 'low'
  }
  return 'default'
}

export function getReasoningControlOptions(
  provider: ModelProvider | undefined,
  model?: ProviderModelInfo | null
): ReasoningControlOption[] {
  const capabilities = getReasoningControlCapabilities(provider, model)
  if (!capabilities.supported) return []

  // OpenAI-style routes expose the extended effort vocabulary. 'off' replaces 'default'
  // here and means "do not send reasoning_effort at all".
  if (capabilities.kind === 'openai-effort') {
    return [{ level: 'off', label: 'off' }, ...OPENAI_REASONING_EFFORT_LEVELS.map((level) => ({ level, label: level }))]
  }

  const offOption: ReasoningControlOption[] = supportsExplicitDisable(
    capabilities.kind,
    getEffectiveProvider(provider, model),
    model?.modelId || ''
  )
    ? [{ level: 'off', label: 'off' }]
    : []

  if (capabilities.kind === 'toggle') {
    return [{ level: 'default', label: 'default' }, ...offOption, { level: 'high', label: 'on' }]
  }

  return [
    { level: 'default', label: 'default' },
    ...offOption,
    { level: 'low', label: 'low' },
    { level: 'medium', label: 'medium' },
    { level: 'high', label: 'high' },
  ]
}

export function getReasoningProviderOptions(
  provider: ModelProvider | undefined,
  model: ProviderModelInfo | null | undefined,
  level: ReasoningControlLevel,
  previous?: ProviderOptions
): ProviderOptions | undefined {
  const capabilities = getReasoningControlCapabilities(provider, model)
  if (!capabilities.supported) return previous

  const effectiveProvider = getEffectiveProvider(provider, model)

  // 'default' means "send no reasoning parameters": drop every reasoning namespace so the
  // provider's server-side default applies. On OpenAI-style routes 'off' carries the same
  // meaning ("do not send reasoning_effort", per the option list), and it is also the
  // fallback for models whose thinking cannot be explicitly disabled.
  if (
    level === 'default' ||
    (level === 'off' &&
      (capabilities.kind === 'openai-effort' ||
        // xAI's Grok reasoning_effort only documents low/high; sending 'none' to disable is not
        // part of the contract, so 'off' means "send no parameter" here as well.
        capabilities.kind === 'xai-effort' ||
        !supportsExplicitDisable(capabilities.kind, effectiveProvider, model?.modelId || '')))
  ) {
    return stripReasoningProviderOptions(previous)
  }

  // ChatboxAI can change a DeepSeek model's apiStyle in the remote catalog.
  // Drop the old protocol namespace before writing the new one so a persisted
  // session never carries Anthropic/OpenAI/Responses controls simultaneously.
  const next: ProviderOptions = isChatboxAIDeepSeekWithOfficialApiStyle(provider, model)
    ? { ...(stripReasoningProviderOptions(previous) || {}) }
    : { ...(previous || {}) }

  if (level === 'off') {
    if (isChatboxAIDeepSeekWithOfficialApiStyle(provider, model)) {
      if (model?.apiStyle === 'anthropic') {
        next.claude = { thinking: { type: 'disabled' } }
      } else if (model?.apiStyle === 'openai-responses') {
        // @ai-sdk/openai does not recognize DeepSeek ids as reasoning models;
        // forceReasoning makes it serialize the official `reasoning.effort` body.
        next.openai = { reasoningEffort: 'none', forceReasoning: true }
      } else {
        next.deepseek = { thinking: { type: 'disabled' } }
      }
    } else if (effectiveProvider === ModelProviderEnum.Claude) {
      next.claude = { thinking: { type: 'disabled', budgetTokens: 0 } }
    } else if (isOpenAICompatibleApiStyle(provider, model as ProviderModelInfo) && isDeepSeekThinkingModel(model)) {
      next.deepseek = { thinking: { type: 'disabled' } }
    } else if (effectiveProvider === ModelProviderEnum.OpenRouter) {
      next.openrouter = { reasoning: { enabled: false, exclude: true } }
    } else if (effectiveProvider === ModelProviderEnum.Gemini) {
      next.google = { thinkingConfig: getGoogleOffThinkingConfig(model?.modelId || '') }
    } else if (effectiveProvider === ModelProviderEnum.DeepSeek) {
      next.deepseek = { thinking: { type: 'disabled' } }
    } else if (effectiveProvider === ModelProviderEnum.Qwen || effectiveProvider === ModelProviderEnum.QwenPortal) {
      next.openaiCompatible = { enable_thinking: false }
    }
    return compactProviderOptions(next)
  }

  // Provider-specific controls only understand the terse low/medium/high levels; extended
  // OpenAI values collapse onto them so a level picked under one model never breaks another.
  const effortLevel = toReasoningEffortLevel(level)

  if (isChatboxAIDeepSeekWithOfficialApiStyle(provider, model)) {
    const effort = isDeepSeekReasoningEffortModel(model?.modelId || '')
      ? DEEPSEEK_EFFORT_BY_LEVEL[effortLevel]
      : undefined
    if (model?.apiStyle === 'anthropic') {
      next.claude = { thinking: { type: 'enabled' }, ...(effort ? { effort } : {}) }
    } else if (model?.apiStyle === 'openai-responses') {
      next.openai = { reasoningEffort: effort, forceReasoning: true }
    } else {
      next.deepseek = { thinking: { type: 'enabled' }, ...(effort ? { reasoningEffort: effort } : {}) }
    }
  } else if (effectiveProvider === ModelProviderEnum.Claude) {
    if (capabilities.kind === 'anthropic-adaptive-effort' || capabilities.kind === 'anthropic-effort') {
      next.claude = { effort: effortLevel }
    } else {
      next.claude = { thinking: { type: 'enabled', budgetTokens: CLAUDE_BUDGET_BY_LEVEL[effortLevel] } }
    }
  } else if (isOpenAICompatibleApiStyle(provider, model as ProviderModelInfo) && isDeepSeekThinkingModel(model)) {
    next.deepseek = {
      thinking: { type: 'enabled' },
      ...(isDeepSeekReasoningEffortModel(model?.modelId || '')
        ? { reasoningEffort: DEEPSEEK_EFFORT_BY_LEVEL[effortLevel] }
        : {}),
    }
  } else if (isOpenAIStyleEffectiveProvider(effectiveProvider)) {
    // Keep compatible wire mappings in pickOpenAICompatibleReasoningOptions in sync when
    // adding an OpenAI option here. OpenAI-only SDK flags must not leak to compatible APIs.
    next.openai = {
      reasoningEffort: getOpenAIReasoningEffort(model?.modelId || '', level),
      ...(effectiveProvider === ModelProviderEnum.OpenAIResponses
        ? {
            reasoningSummary: 'auto' as const,
            include: ['reasoning.encrypted_content'],
            forceReasoning: true,
          }
        : {}),
    }
  } else if (
    capabilities.kind === 'openai-effort' &&
    isOpenAICompatibleApiStyle(provider, model as ProviderModelInfo)
  ) {
    // Generic OpenAI-compatible model with no stamped apiStyle to route on: @ai-sdk/
    // openai-compatible forwards `reasoningEffort` verbatim as the `reasoning_effort` body field.
    next.openaiCompatible = { reasoningEffort: level }
  } else if (effectiveProvider === ModelProviderEnum.XAI) {
    next.openai = {
      reasoningEffort: effortLevel,
      include: ['reasoning.encrypted_content'],
      forceReasoning: true,
    }
  } else if (effectiveProvider === ModelProviderEnum.OpenRouter) {
    next.openrouter = {
      reasoning: {
        effort: effortLevel,
        exclude: false,
      },
    }
  } else if (effectiveProvider === ModelProviderEnum.Gemini) {
    if (capabilities.kind === 'level') {
      next.google = {
        thinkingConfig: { thinkingLevel: effortLevel as GoogleThinkingLevel, includeThoughts: true },
      }
    } else {
      next.google = {
        thinkingConfig: { thinkingBudget: GEMINI_BUDGET_BY_LEVEL[effortLevel], includeThoughts: true },
      }
    }
  } else if (effectiveProvider === ModelProviderEnum.DeepSeek) {
    next.deepseek = {
      thinking: { type: 'enabled' },
      ...(isDeepSeekReasoningEffortModel(model?.modelId || '')
        ? { reasoningEffort: DEEPSEEK_EFFORT_BY_LEVEL[effortLevel] }
        : {}),
    }
  } else if (effectiveProvider === ModelProviderEnum.Qwen || effectiveProvider === ModelProviderEnum.QwenPortal) {
    next.openaiCompatible = {
      enable_thinking: true,
      thinking_budget: QWEN_THINKING_BUDGET_BY_LEVEL[effortLevel],
    }
  }

  return compactProviderOptions(next)
}

function isDeepSeekThinkingModel(model: ProviderModelInfo | null | undefined): boolean {
  return !!model?.modelId && isDeepSeekReasoningModel(model.modelId)
}

function isChatboxAIDeepSeekWithOfficialApiStyle(
  provider: ModelProvider | undefined,
  model: ProviderModelInfo | null | undefined
): boolean {
  return (
    provider === ModelProviderEnum.ChatboxAI &&
    !!model &&
    isDeepSeekReasoningModel(model.modelId) &&
    (!model.apiStyle ||
      model.apiStyle === 'openai' ||
      model.apiStyle === 'anthropic' ||
      (model.apiStyle === 'openai-responses' && isDeepSeekReasoningEffortModel(model.modelId)))
  )
}

function deriveDeepSeekLevel(
  modelId: string,
  thinkingType: 'enabled' | 'disabled' | undefined,
  effort: string | undefined
): ReasoningControlLevel {
  if (thinkingType === 'disabled') return 'off'
  if (!isDeepSeekReasoningEffortModel(modelId)) return thinkingType === 'enabled' ? 'high' : 'default'
  if (thinkingType !== 'enabled' && !effort) return 'default'
  return normalizeDeepSeekEffortToLevel(effort)
}

function deriveDeepSeekResponsesLevel(modelId: string, effort: string | undefined): ReasoningControlLevel {
  if (!isDeepSeekReasoningEffortModel(modelId) || !effort) return 'default'
  if (effort === 'none') return 'off'
  return normalizeDeepSeekEffortToLevel(effort)
}

function normalizeDeepSeekEffortToLevel(effort: string | undefined): ReasoningControlLevel {
  if (effort === 'low') return 'low'
  if (effort === 'high') return 'medium'
  if (effort === 'max' || effort === 'xhigh') return 'high'
  return 'default'
}

function getGoogleOffThinkingConfig(modelId: string): NonNullable<ProviderOptions['google']>['thinkingConfig'] {
  if (getGoogleThinkingMode(modelId) === 'level') {
    const supportedLevels = getSupportedGoogleThinkingLevels(modelId)
    return {
      thinkingLevel: supportedLevels.includes('minimal') ? 'minimal' : 'low',
      includeThoughts: false,
    }
  }

  return { thinkingBudget: 0, includeThoughts: false }
}

function isOpenRouterReasoningModel(model: ProviderModelInfo | null | undefined): boolean {
  if (!model?.modelId) return false
  if (isDeepSeekReasoningModel(model.modelId)) return true
  if (isGptEffortModel(model.modelId)) return true
  return matchesAny(model.modelId, [
    ...CLAUDE_ADAPTIVE_EFFORT_MODELS,
    ...CLAUDE_EFFORT_MODELS,
    ...CLAUDE_BUDGET_MODELS,
    ...QWEN_THINKING_MODELS,
    ...GROK_REASONING_EFFORT_MODELS,
    // o1-preview/o1-mini reject the direct reasoning_effort param, but OpenRouter maps
    // its own reasoning options per model, so they keep reasoning controls there.
    ...OPENAI_NO_EFFORT_PARAM_MODELS,
  ])
}

/**
 * The reasoning_effort value forwarded for a selected level. Every extended level is sent
 * verbatim so OpenAI-compatible endpoints can accept values beyond OpenAI's own set.
 * `off`/`default` never reach this function — they strip the parameter instead.
 */
export function getOpenAIReasoningEffort(
  modelId: string,
  level: Exclude<ReasoningControlLevel, 'default' | 'off'>
): NonNullable<ProviderOptions['openai']>['reasoningEffort'] {
  // o1-preview/o1-mini reject the parameter entirely; the capability gate keeps them out of
  // the UI, so this guard only protects stale options. Keep the model id in the signature
  // because callers pass the model they are about to write options for.
  void modelId
  return level
}

/** Readback for OpenAI-style routes: no stored effort means 'off' (nothing is sent). */
function normalizeOpenAIEffortToLevel(effort: string | undefined): ReasoningControlLevel {
  if (!effort) return 'off'
  return (OPENAI_REASONING_EFFORT_LEVELS as readonly string[]).includes(effort)
    ? (effort as OpenAIReasoningEffortLevel)
    : 'high'
}

function normalizeEffortToLevel(effort: string | undefined): ReasoningControlLevel {
  if (!effort) return 'default'
  if (effort === 'none' || effort === 'minimal') return 'off'
  if (effort === 'low' || effort === 'medium' || effort === 'high') return effort
  return 'high'
}

function compactProviderOptions(options: ProviderOptions): ProviderOptions | undefined {
  const next: ProviderOptions = { ...options }
  if (!next.claude) delete next.claude
  if (!next.openai) delete next.openai
  if (!next.google) delete next.google
  if (!next.deepseek) delete next.deepseek
  if (!next.openaiCompatible) delete next.openaiCompatible
  if (!next.openrouter) delete next.openrouter
  return Object.keys(next).length > 0 ? next : undefined
}

// Provider option namespaces that exclusively carry reasoning/thinking configuration.
// Keep this in sync with ProviderOptionsSchema in shared/types/settings.ts.
const REASONING_PROVIDER_OPTION_KEYS = [
  'claude',
  'openai',
  'google',
  'deepseek',
  'openaiCompatible',
  'openrouter',
] as const satisfies readonly (keyof ProviderOptions)[]

/**
 * Removes reasoning/thinking provider options so they are never sent to a model
 * that does not support reasoning control. This guards against stale options
 * persisted on a session (e.g. set on a reasoning-capable model, then carried
 * over after switching to a model without reasoning support). It also implements
 * the 'default' reasoning level (send no reasoning parameters) and the fallback
 * when 'off' is requested for a model whose thinking cannot be force-disabled.
 */
export function stripReasoningProviderOptions(
  providerOptions: ProviderOptions | undefined
): ProviderOptions | undefined {
  if (!providerOptions) return providerOptions
  const next: ProviderOptions = { ...providerOptions }
  let changed = false
  for (const key of REASONING_PROVIDER_OPTION_KEYS) {
    if (next[key] !== undefined) {
      delete next[key]
      changed = true
    }
  }
  if (!changed) return providerOptions
  return Object.keys(next).length > 0 ? next : undefined
}
