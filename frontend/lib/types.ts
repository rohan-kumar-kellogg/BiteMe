// API Types for Flavor Fingerprint

export interface TasteDimension {
  key: string
  label: string
  value: number
  description?: string
}

export interface TasteProfile {
  dimensions: TasteDimension[]
  analysis: string
  relative_rankings: RelativeRanking[]
  history: TasteTrendPoint[]
}

export interface TasteTrendPoint {
  timestamp: string
  upload_count: number
  interaction_count?: number
  dimensions: {
    sweet_leaning: number
    spicy_leaning: number
    richness_preference: number
    freshness_preference: number
    dessert_affinity: number
  }
}

export interface RelativeRanking {
  label: string
  description: string
  percentile: number
  direction: 'higher' | 'lower' | 'similar'
}

export interface CompatibleUser {
  username: string
  email?: string
  compatibility_score: number
  reason: string
  avatar?: string
}

export interface InvitePayload {
  to_username: string
  to_email?: string
  restaurant_name?: string
  date?: string
  time?: string
  message?: string
}

export interface RestaurantAction {
  type: 'resy' | 'opentable' | 'website' | 'call' | 'none'
  label: string
  url: string
}

export interface RestaurantRecommendation {
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
  source: string
  compatibility_score: number
  explanation: string
  booking_action: RestaurantAction
  action: {
    primary: {
      type: 'book' | 'website' | 'call' | 'none' | string
      label: string
      target: string
    }
    reservation_provider?: string
    reservation_url?: string
    website_url?: string
    phone?: string
  }
  score_breakdown: {
    cuisine_match: { score: number; weight: number; weighted: number }
    trait_match: { score: number; weight: number; weighted: number }
    dish_match: { score: number; weight: number; weighted: number }
    location_match: { score: number; weight: number; weighted: number }
    reservation_match: { score: number; weight: number; weighted: number }
    popularity_match: { score: number; weight: number; weighted: number }
  }
}

export interface RecentUpload {
  id: string
  image_url: string
  dish_label: string
  cuisine?: string
  tags?: string[]
  uploaded_at: string
}

export interface ClickedRecommendation {
  event_id: string
  dish_label: string
  cuisine: string
  signal_weight: number
  timestamp: string
}

export interface UserProfile {
  username: string
  email: string
  archetype: string
  archetype_description: string
  archetype_graphic?: string
  joke: string
  observations: string[]
  favorite_cuisines: Record<string, number>
  favorite_dishes: Record<string, number>
  favorite_traits: Record<string, number>
  taste_profile: TasteProfile
  compatible_users: CompatibleUser[]
  recent_uploads: RecentUpload[]
  clicked_recommendations: ClickedRecommendation[]
  total_uploads: number
  created_at: string
}

export interface LoadOrCreateResult {
  user: UserProfile
  isNew: boolean
}

export interface SessionIdentity {
  username: string
  email: string
}

export type AppState = 'welcome' | 'loading' | 'profile' | 'new-user' | 'error'
