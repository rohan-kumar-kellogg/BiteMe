'use client'

import { useState } from 'react'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { CheckCircle2, Crown, Sparkles } from 'lucide-react'
import type { UserProfile } from '@/lib/types'

interface ProfileHeroProps {
  profile: UserProfile
  onRecommendationClick: (dishLabel: string, cuisine?: string) => Promise<void>
}

interface RecommendationItem {
  dish: string
  cuisine: string
}

export function ProfileHero({ profile, onRecommendationClick }: ProfileHeroProps) {
  const hasUploads = profile.total_uploads > 0
  const [activeDish, setActiveDish] = useState<string | null>(null)
  const [recentlyAddedDish, setRecentlyAddedDish] = useState<string | null>(null)
  const [feedbackMessage, setFeedbackMessage] = useState('')
  const recommendations = buildRecommendations(profile, 5)
  const handleRecommendationClick = async (item: RecommendationItem) => {
    if (activeDish || recentlyAddedDish === item.dish) return
    try {
      setActiveDish(item.dish)
      await onRecommendationClick(item.dish, item.cuisine)
      setRecentlyAddedDish(item.dish)
      setFeedbackMessage('Taste profile updated')
      window.setTimeout(() => {
        setRecentlyAddedDish((prev) => (prev === item.dish ? null : prev))
        setFeedbackMessage('')
      }, 1800)
    } finally {
      setActiveDish(null)
    }
  }
  return (
    <section className="relative overflow-hidden rounded-3xl bg-card border border-border/50">
      {/* Background Pattern */}
      <div className="absolute inset-0 opacity-5">
        <div className="absolute inset-0 bg-[radial-gradient(circle_at_30%_40%,var(--primary)_0%,transparent_50%)]" />
        <div className="absolute inset-0 bg-[radial-gradient(circle_at_70%_80%,var(--primary)_0%,transparent_40%)]" />
      </div>
      
      <div className="relative p-6 md:p-10 space-y-6">
        {/* Top Row: Username & Badge */}
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-14 h-14 rounded-2xl bg-primary/10 flex items-center justify-center">
              <span className="text-2xl font-semibold text-primary">
                {profile.username.charAt(0).toUpperCase()}
              </span>
            </div>
            <div>
              <h1 className="text-xl font-semibold">@{profile.username}</h1>
              <p className="text-sm text-muted-foreground">
                {profile.total_uploads} dishes explored
              </p>
            </div>
          </div>
          <Badge variant="secondary" className="rounded-full px-3 py-1 gap-1.5">
            <Crown className="w-3.5 h-3.5" />
            <span className="text-xs font-medium">Level {Math.floor(profile.total_uploads / 10) + 1}</span>
          </Badge>
        </div>

        {/* Archetype Display */}
        <div className="space-y-4 pt-2">
          <div className="flex items-center gap-2">
            <Sparkles className="w-5 h-5 text-primary" />
            <span className="text-sm font-medium text-primary uppercase tracking-wider">Your Archetype</span>
          </div>
          
          <h2 className="text-3xl md:text-4xl font-semibold tracking-tight text-balance">
            {hasUploads ? (profile.archetype || 'The Explorer') : 'Your profile is warming up'}
          </h2>
          
          <p className="text-muted-foreground leading-relaxed text-pretty max-w-2xl">
            {hasUploads
              ? (profile.archetype_description || 'Your taste profile is still forming. Upload more dishes to discover your unique culinary identity.')
              : 'Upload your first few dishes and BiteMe will start turning your meals into a personalized flavor fingerprint.'}
          </p>

          {hasUploads && recommendations.length > 0 && (
            <div className="pt-2">
              <p className="text-sm font-medium text-foreground">You May Also Like:</p>
              <div className="mt-2 flex flex-wrap gap-2">
                {recommendations.map((item) => {
                  const isSaving = activeDish === item.dish
                  const isAdded = recentlyAddedDish === item.dish
                  return (
                  <Button
                    key={item.dish}
                    size="sm"
                    variant={isAdded || isSaving ? 'default' : 'secondary'}
                    className="rounded-full h-7 px-3 text-xs transition-all"
                    disabled={Boolean(activeDish) || isAdded}
                    onClick={() => handleRecommendationClick(item)}
                  >
                    {isSaving ? 'Saving...' : isAdded ? (
                      <span className="inline-flex items-center gap-1">
                        <CheckCircle2 className="w-3 h-3" />
                        Added
                      </span>
                    ) : item.dish}
                  </Button>
                  )
                })}
              </div>
              {feedbackMessage ? (
                <p className="mt-2 text-xs text-primary">{feedbackMessage}</p>
              ) : null}
            </div>
          )}
        </div>
      </div>
    </section>
  )
}

function titleize(x: string): string {
  return x
    .replace(/_/g, ' ')
    .replace(/\b\w/g, (m) => m.toUpperCase())
}

function buildRecommendations(profile: UserProfile, maxItems: number): RecommendationItem[] {
  const seen = new Set<string>()
  for (const upload of profile.recent_uploads) {
    seen.add(String(upload.dish_label || '').trim().toLowerCase())
  }
  for (const dish of Object.keys(profile.favorite_dishes || {})) {
    seen.add(dish.trim().toLowerCase())
  }

  const suggestions: RecommendationItem[] = []
  const add = (dish: string, cuisine = '') => {
    const key = dish.trim().toLowerCase()
    if (!key || seen.has(key) || suggestions.some((x) => x.dish.toLowerCase() === key)) return
    suggestions.push({ dish: titleize(dish), cuisine })
  }

  const topCuisines = Object.entries(profile.favorite_cuisines || {})
    .sort((a, b) => b[1] - a[1])
    .map(([name]) => name.toLowerCase())
  const topTraits = Object.entries(profile.favorite_traits || {})
    .sort((a, b) => b[1] - a[1])
    .map(([name]) => name.toLowerCase())

  const byCuisine: Record<string, string[]> = {
    japanese: ['tonkotsu ramen', 'yakitori bowl', 'unagi don'],
    italian: ['cacio e pepe', 'eggplant parmigiana', 'mushroom risotto'],
    mexican: ['al pastor tacos', 'chilaquiles', 'birria quesadilla'],
    indian: ['paneer tikka masala', 'chicken korma', 'masala dosa'],
    thai: ['pad kee mao', 'khao soi', 'tom yum noodles'],
    korean: ['bibimbap', 'dakgalbi', 'kimchi fried rice'],
    american: ['smash burger', 'hot chicken sandwich', 'loaded baked potato'],
    mediterranean: ['chicken shawarma plate', 'falafel pita', 'mujadara bowl'],
    french: ['coq au vin', 'onion gratin soup', 'croque monsieur'],
  }
  const byTrait: Array<[string, string[]]> = [
    ['dessert-leaning', ['chocolate lava cake', 'banoffee pie', 'pistachio gelato']],
    ['spice-forward', ['thai basil chicken', 'sichuan dan dan noodles', 'peri peri chicken wrap']],
    ['comfort-food', ['baked mac and cheese', 'chicken pot pie', 'beef stew']],
    ['protein-forward', ['grilled steak bowl', 'salmon rice bowl', 'chicken souvlaki plate']],
    ['carb-forward', ['truffle mushroom pasta', 'garlic butter noodles', 'crispy potato tacos']],
    ['plant-forward', ['roasted veggie grain bowl', 'caprese salad', 'tofu bibimbap']],
  ]

  for (const c of topCuisines) {
    for (const dish of byCuisine[c] || []) add(dish, titleize(c))
    if (suggestions.length >= maxItems) return suggestions.slice(0, maxItems)
  }
  for (const [trait, dishes] of byTrait) {
    if (topTraits.includes(trait)) {
      for (const dish of dishes) add(dish)
      if (suggestions.length >= maxItems) return suggestions.slice(0, maxItems)
    }
  }

  for (const dish of ['ramen', 'sushi', 'grilled fish tacos', 'mushroom risotto', 'tiramisu']) {
    add(dish)
    if (suggestions.length >= maxItems) break
  }
  return suggestions.slice(0, maxItems)
}
