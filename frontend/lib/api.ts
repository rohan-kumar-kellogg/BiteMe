import type {
  ClickedRecommendation,
  CompatibleUser,
  InvitePayload,
  LoadOrCreateResult,
  RecentUpload,
  RestaurantRecommendation,
  RelativeRanking,
  TasteDimension,
  TasteTrendPoint,
  UserProfile,
} from './types'

const API_BASE_URL = (process.env.NEXT_PUBLIC_API_BASE_URL || 'http://127.0.0.1:8000').replace(/\/$/, '')
const API_BASE = `${API_BASE_URL}/api`

type ApiCompatibleUser = {
  compatible_username: string
  compatible_email?: string
  compatibility_score: number
  why_you_match?: string
}

type ApiRelativeRanking = {
  dimension: string
  percentile_within_user_base: number
  interpretation: string
}

type ApiUserPayload = {
  username: string
  email?: string
  created_at: string
  archetype: string
  archetype_description: string
  archetype_graphic?: string
  observations: string | string[]
  joke: string
  profile?: {
    upload_count?: number
    favorite_cuisines?: Record<string, number>
    favorite_dishes?: Record<string, number>
    favorite_traits?: Record<string, number>
    recommendation_feedback?: Array<{
      event_id?: string
      event_type?: string
      dish_label?: string
      cuisine?: string
      signal_weight?: number
      timestamp?: string
    }>
    taste_profile?: {
      dimensions?: Record<string, { score?: number; explanation?: string }>
      analysis?: unknown
      relative_rankings?: ApiRelativeRanking[]
      history?: Array<{
        timestamp?: string
        upload_count?: number
        dimensions?: {
          sweet_leaning?: number
          spicy_leaning?: number
          richness_preference?: number
          freshness_preference?: number
          dessert_affinity?: number
        }
      }>
    }
  }
  compatible_users?: ApiCompatibleUser[]
}

type ApiUpload = {
  id: number | string
  created_at: string
  image_path?: string
  prediction?: {
    predicted_label?: string
    top3_candidates?: Array<{ dish_label?: string; cuisine?: string }>
  }
}

type ApiUploadResponse = {
  status: string
  upload_id?: string | number
  prediction?: {
    predicted_label?: string
    abstained?: boolean
    rejected_as_not_food?: boolean
  }
  upload_debug?: unknown
}

type ApiRestaurantRecommendation = {
  restaurant: {
    id: string
    name: string
    address?: string
    zip_code: string
    latitude?: number
    longitude?: number
    cuisine_tags: string[]
    menu_tags: string[]
    trait_tags: string[]
    rating: number
    review_count: number
    venue_type: string
    service_type?: string
    source?: string
  }
  compatibility_score: number
  explanation: string
  booking_action?: {
    type?: 'resy' | 'opentable' | 'website' | 'call' | 'none'
    label?: string
    url?: string
  }
  action: {
    primary: { type: 'book' | 'website' | 'call' | 'none'; label: string; target: string }
    reservation_provider?: string
    reservation_url?: string
    website_url?: string
    phone?: string
  }
  score_breakdown: RestaurantRecommendation['score_breakdown']
}

function toLabel(key: string): string {
  return key
    .replace(/_/g, ' ')
    .replace(/\b\w/g, (s) => s.toUpperCase())
}

function toPercent(x: number): number {
  return Math.max(0, Math.min(100, Math.round(x * 100)))
}

function mapCompatibleUsers(rows: ApiCompatibleUser[] | undefined): CompatibleUser[] {
  if (!rows || rows.length === 0) return []
  return rows.map((u) => ({
    username: u.compatible_username,
    email: u.compatible_email || '',
    compatibility_score: toPercent(Number(u.compatibility_score || 0)),
    reason: u.why_you_match || 'Your food habits overlap in a way that feels natural.',
  }))
}

function mapRelativeRankings(rows: ApiRelativeRanking[] | undefined): RelativeRanking[] {
  if (!rows || rows.length === 0) return []
  return rows.map((r) => {
    const p = Math.max(0, Math.min(100, Math.round(Number(r.percentile_within_user_base || 0))))
    let direction: RelativeRanking['direction'] = 'similar'
    if (p >= 60) direction = 'higher'
    else if (p <= 40) direction = 'lower'
    return {
      label: toLabel(r.dimension),
      description: r.interpretation,
      percentile: p,
      direction,
    }
  })
}

function mapTasteDimensions(
  dims: Record<string, { score?: number; explanation?: string }> | undefined
): TasteDimension[] {
  if (!dims) return []
  return Object.entries(dims).map(([key, info]) => ({
    key,
    label: toLabel(key),
    value: toPercent(Number(info?.score || 0)),
    description: info?.explanation || '',
  }))
}

function mapAnalysis(analysis: unknown): string {
  if (!analysis) return ''
  if (typeof analysis === 'string') return analysis
  if (typeof analysis !== 'object') return ''
  const a = analysis as Record<string, unknown>
  const likes = Array.isArray(a.likes) ? a.likes.slice(0, 3).join(', ') : ''
  const less = Array.isArray(a.less_affinity_for) ? a.less_affinity_for.slice(0, 2).join(', ') : ''
  const patterns = Array.isArray(a.notable_patterns) ? a.notable_patterns.slice(0, 2).join(' ') : ''
  const parts = [
    likes ? `Strong likes: ${likes}.` : '',
    less ? `Lower affinity: ${less}.` : '',
    patterns || '',
  ].filter(Boolean)
  return parts.join(' ')
}

function mapObservations(observations: string | string[] | undefined): string[] {
  if (!observations) return []
  if (Array.isArray(observations)) return observations.map((x) => String(x)).filter(Boolean)
  return String(observations)
    .split('\n')
    .map((x) => x.trim())
    .filter(Boolean)
}

function mapRecentUploads(rows: ApiUpload[] | undefined): RecentUpload[] {
  if (!rows || rows.length === 0) return []
  return rows.map((u) => ({
    id: String(u.id),
    image_url: u.image_path || '',
    dish_label: u.prediction?.predicted_label || 'Unknown Dish',
    cuisine: u.prediction?.top3_candidates?.[0]?.cuisine || 'Unknown',
    tags: [],
    uploaded_at: u.created_at,
  }))
}

function mapTrendHistory(rows: Array<{
  timestamp?: string
  upload_count?: number
  interaction_count?: number
  dimensions?: {
    sweet_leaning?: number
    spicy_leaning?: number
    richness_preference?: number
    freshness_preference?: number
    dessert_affinity?: number
  }
}> | undefined): TasteTrendPoint[] {
  if (!Array.isArray(rows)) return []
  return rows
    .map((x) => ({
      timestamp: String(x?.timestamp || ''),
      upload_count: Number(x?.upload_count || 0),
      interaction_count: Number(x?.interaction_count || 0),
      dimensions: {
        sweet_leaning: toPercent(Number(x?.dimensions?.sweet_leaning || 0)),
        spicy_leaning: toPercent(Number(x?.dimensions?.spicy_leaning || 0)),
        richness_preference: toPercent(Number(x?.dimensions?.richness_preference || 0)),
        freshness_preference: toPercent(Number(x?.dimensions?.freshness_preference || 0)),
        dessert_affinity: toPercent(Number(x?.dimensions?.dessert_affinity || 0)),
      },
    }))
    .filter((x) => Boolean(x.timestamp))
}

function mapClickedRecommendations(
  rows: Array<{
    event_id?: string
    event_type?: string
    dish_label?: string
    cuisine?: string
    signal_weight?: number
    timestamp?: string
  }> | undefined
): ClickedRecommendation[] {
  if (!Array.isArray(rows)) return []
  return rows
    .filter((x) => String(x?.event_type || '') === 'recommendation_click')
    .map((x, i) => ({
      event_id: String(x?.event_id || `${x?.timestamp || 'evt'}_${i}`),
      dish_label: String(x?.dish_label || '').trim(),
      cuisine: String(x?.cuisine || '').trim(),
      signal_weight: Number(x?.signal_weight || 0),
      timestamp: String(x?.timestamp || ''),
    }))
    .filter((x) => Boolean(x.dish_label))
}

function mapUserPayload(user: ApiUserPayload, recentUploads: ApiUpload[] = []): UserProfile {
  const tasteProfile = user.profile?.taste_profile
  const dimensions = mapTasteDimensions(tasteProfile?.dimensions)
  const relativeRankings = mapRelativeRankings(tasteProfile?.relative_rankings)
  const uploads = mapRecentUploads(recentUploads)
  const uploadCount = Number(user.profile?.upload_count || uploads.length || 0)
  return {
    username: user.username,
    email: user.email || '',
    created_at: user.created_at,
    archetype: user.archetype || '',
    archetype_description: user.archetype_description || '',
    archetype_graphic: user.archetype_graphic || '',
    joke: user.joke || '',
    observations: mapObservations(user.observations),
    favorite_cuisines: user.profile?.favorite_cuisines || {},
    favorite_dishes: user.profile?.favorite_dishes || {},
    favorite_traits: user.profile?.favorite_traits || {},
    taste_profile: {
      dimensions,
      analysis: mapAnalysis(tasteProfile?.analysis),
      relative_rankings: relativeRankings,
      history: mapTrendHistory(tasteProfile?.history),
    },
    compatible_users: mapCompatibleUsers(user.compatible_users),
    recent_uploads: uploads,
    clicked_recommendations: mapClickedRecommendations(user.profile?.recommendation_feedback),
    total_uploads: uploadCount,
  }
}

async function parseJson<T>(response: Response): Promise<T> {
  if (!response.ok) {
    let detail = ''
    try {
      const payload = await response.json()
      detail = payload?.detail ? `: ${payload.detail}` : ''
    } catch {
      // ignore parsing error and fallback to status text
    }
    throw new Error(`Request failed (${response.status})${detail || `: ${response.statusText}`}`)
  }
  return response.json() as Promise<T>
}

async function fetchWithTimeout(input: RequestInfo | URL, init: RequestInit = {}, timeoutMs = 30000): Promise<Response> {
  const controller = new AbortController()
  const timer = window.setTimeout(() => controller.abort(), timeoutMs)
  try {
    return await fetch(input, { ...init, signal: controller.signal })
  } catch (err) {
    if (err instanceof DOMException && err.name === 'AbortError') {
      throw new Error(`Request timed out after ${Math.round(timeoutMs / 1000)}s`)
    }
    throw err
  } finally {
    window.clearTimeout(timer)
  }
}

export async function loadOrCreateUser(username: string, email: string): Promise<LoadOrCreateResult> {
  console.info('[api] loadOrCreateUser start', { username })
  try {
    const response = await fetchWithTimeout(
      `${API_BASE}/users/load_or_create`,
      {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ username, email }),
      },
      30000
    )
    const payload = await parseJson<{ created: boolean; user: ApiUserPayload }>(response)
    console.info('[api] loadOrCreateUser success', { username, created: Boolean(payload.created) })
    return {
      user: mapUserPayload(payload.user),
      isNew: Boolean(payload.created),
    }
  } catch (error) {
    console.error('[api] loadOrCreateUser failed', { username, error })
    throw error
  }
}

export async function getUserProfile(username: string): Promise<UserProfile> {
  console.info('[api] getUserProfile start', { username })
  try {
    const response = await fetchWithTimeout(`${API_BASE}/users/${encodeURIComponent(username)}`, {}, 30000)
    const payload = await parseJson<{ user: ApiUserPayload; recent_uploads: ApiUpload[] }>(response)
    console.info('[api] getUserProfile success', { username, uploads: payload?.recent_uploads?.length || 0 })
    return mapUserPayload(payload.user, payload.recent_uploads)
  } catch (error) {
    console.error('[api] getUserProfile failed', { username, error })
    throw error
  }
}

export async function getCompatibleUsers(username: string): Promise<CompatibleUser[]> {
  console.info('[api] getCompatibleUsers start', { username })
  try {
    const response = await fetchWithTimeout(`${API_BASE}/users/${encodeURIComponent(username)}/compatible`, {}, 30000)
    const payload = await parseJson<{ compatible_users: ApiCompatibleUser[] }>(response)
    console.info('[api] getCompatibleUsers success', { username, count: payload?.compatible_users?.length || 0 })
    return mapCompatibleUsers(payload.compatible_users)
  } catch (error) {
    console.error('[api] getCompatibleUsers failed', { username, error })
    throw error
  }
}

export async function uploadUserImage(username: string, file: File): Promise<ApiUploadResponse> {
  console.info('[api] uploadUserImage start', { username, file: file?.name || 'unknown', size: file?.size || 0 })
  const formData = new FormData()
  formData.append('image', file)
  const url = `${API_BASE}/users/${encodeURIComponent(username)}/uploads?debug_metadata=true`
  try {
    const response = await fetchWithTimeout(
      url,
      {
        method: 'POST',
        body: formData,
      },
      60000
    )
    const payload = await parseJson<ApiUploadResponse>(response)
    console.info('[api] uploadUserImage success', {
      username,
      predicted_label: payload?.prediction?.predicted_label || '',
      abstained: Boolean(payload?.prediction?.abstained),
      rejected_as_not_food: Boolean(payload?.prediction?.rejected_as_not_food),
    })
    return payload
  } catch (error) {
    console.error('[api] uploadUserImage failed', { username, error })
    throw error
  }
}

export async function sendRecommendationFeedback(
  username: string,
  dishLabel: string,
  cuisine = ''
): Promise<void> {
  const response = await fetch(`${API_BASE}/users/${encodeURIComponent(username)}/recommendations/feedback`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ dish_label: dishLabel, cuisine }),
  })
  await parseJson(response)
}

export async function removeRecommendationFeedback(username: string, eventId: string): Promise<void> {
  const response = await fetch(
    `${API_BASE}/users/${encodeURIComponent(username)}/recommendations/feedback/${encodeURIComponent(eventId)}`,
    { method: 'DELETE' }
  )
  await parseJson(response)
}

export async function removeUserUpload(username: string, uploadId: string): Promise<void> {
  const response = await fetch(
    `${API_BASE}/users/${encodeURIComponent(username)}/uploads/${encodeURIComponent(uploadId)}`,
    { method: 'DELETE' }
  )
  await parseJson(response)
}

export async function getRestaurantRecommendations(
  username: string,
  zipCode: string,
  limit = 20,
  context = ''
): Promise<RestaurantRecommendation[]> {
  const params = new URLSearchParams({ zip_code: zipCode, limit: String(limit) })
  if (context) params.set('context', context)
  const response = await fetch(`${API_BASE}/users/${encodeURIComponent(username)}/restaurants?${params.toString()}`)
  const payload = await parseJson<{ restaurants: ApiRestaurantRecommendation[] }>(response)
  return (payload.restaurants || []).map((row) => ({
    id: row.restaurant.id,
    name: row.restaurant.name,
    address: row.restaurant.address || '',
    zip_code: row.restaurant.zip_code,
    latitude: row.restaurant.latitude,
    longitude: row.restaurant.longitude,
    cuisine_tags: row.restaurant.cuisine_tags || [],
    menu_tags: row.restaurant.menu_tags || [],
    trait_tags: row.restaurant.trait_tags || [],
    rating: Number(row.restaurant.rating || 0),
    review_count: Number(row.restaurant.review_count || 0),
    venue_type: row.restaurant.venue_type || '',
    service_type: row.restaurant.service_type || '',
    source: row.restaurant.source || 'unknown',
    compatibility_score: Number(row.compatibility_score || 0),
    explanation: row.explanation || '',
    booking_action: {
      type: row.booking_action?.type || 'none',
      label: row.booking_action?.label || 'No action available',
      url: row.booking_action?.url || '',
    },
    action: row.action || { primary: { type: 'none', label: 'No action available', target: '' } },
    score_breakdown: row.score_breakdown,
  }))
}

export async function createInvite(
  username: string,
  payload: InvitePayload
): Promise<void> {
  const response = await fetch(`${API_BASE}/users/${encodeURIComponent(username)}/invites`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  })
  await parseJson(response)
}
