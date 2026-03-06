'use client'

import { useEffect, useState } from 'react'
import { CheckCircle2, Users, Heart, Sparkles } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { Textarea } from '@/components/ui/textarea'
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog'
import { getRestaurantRecommendations } from '@/lib/api'
import type { CompatibleUser, RestaurantRecommendation } from '@/lib/types'

interface CompatibleUsersProps {
  users: CompatibleUser[]
  currentUsername: string
  onInviteToEat: (payload: {
    to_username: string
    to_email?: string
    restaurant_name?: string
    date?: string
    time?: string
    message?: string
  }) => Promise<void>
}

export function CompatibleUsers({ users, currentUsername, onInviteToEat }: CompatibleUsersProps) {
  const [zip, setZip] = useState('')
  const [activeTab, setActiveTab] = useState<'taste' | 'restaurant'>('taste')
  const [restaurants, setRestaurants] = useState<RestaurantRecommendation[]>([])
  const [isLoadingRestaurants, setIsLoadingRestaurants] = useState(false)
  const [restaurantError, setRestaurantError] = useState('')
  const [inviteOpen, setInviteOpen] = useState(false)
  const [inviteTarget, setInviteTarget] = useState<CompatibleUser | null>(null)
  const [inviteRestaurant, setInviteRestaurant] = useState('')
  const [inviteDate, setInviteDate] = useState('')
  const [inviteTime, setInviteTime] = useState('')
  const [inviteMessage, setInviteMessage] = useState('')
  const [isSendingInvite, setIsSendingInvite] = useState(false)
  const [inviteErrors, setInviteErrors] = useState<{ restaurant?: string; date?: string; time?: string }>({})
  const [inviteSuccessMessage, setInviteSuccessMessage] = useState('')
  const [inviteContextRestaurant, setInviteContextRestaurant] = useState('')

  useEffect(() => {
    let cancelled = false
    const run = async () => {
      if (activeTab !== 'restaurant') return
      if (zip.length !== 5) {
        setRestaurants([])
        setRestaurantError('')
        return
      }
      setIsLoadingRestaurants(true)
      setRestaurantError('')
      try {
        const rows = await getRestaurantRecommendations(currentUsername, zip, 8)
        if (!cancelled) setRestaurants(rows)
      } catch {
        if (!cancelled) {
          setRestaurants([])
          setRestaurantError('Could not fetch restaurant matches right now.')
        }
      } finally {
        if (!cancelled) setIsLoadingRestaurants(false)
      }
    }
    void run()
    return () => {
      cancelled = true
    }
  }, [activeTab, zip, currentUsername])

  const selectedRestaurant = restaurants[0]?.name || ''
  const openInviteModal = (target: CompatibleUser) => {
    setInviteTarget(target)
    setInviteRestaurant(selectedRestaurant || '')
    setInviteContextRestaurant(selectedRestaurant || '')
    setInviteMessage('')
    setInviteDate('')
    setInviteTime('')
    setInviteErrors({})
    setInviteOpen(true)
  }
  const buildMailto = (target: CompatibleUser, useModalContext = false) => {
    const restaurant = useModalContext ? inviteRestaurant : selectedRestaurant
    const when = [inviteDate, inviteTime].filter(Boolean).join(' ')
    const subject = restaurant
      ? `Invite to eat at ${restaurant}`
      : `Invite to eat`
    const body = [
      `Hey ${target.username},`,
      '',
      restaurant ? `Want to eat at ${restaurant}?` : 'Want to eat soon?',
      when ? `When: ${when}` : '',
      inviteMessage ? `Note: ${inviteMessage}` : '',
      '',
      `- ${currentUsername}`,
    ].filter(Boolean).join('\n')
    const email = (target.email || '').trim()
    return `mailto:${encodeURIComponent(email)}?subject=${encodeURIComponent(subject)}&body=${encodeURIComponent(body)}`
  }
  const handleSendInvite = async () => {
    if (!inviteTarget) return
    const nextErrors: { restaurant?: string; date?: string; time?: string } = {}
    if (!inviteRestaurant.trim()) nextErrors.restaurant = 'Choose or enter a restaurant.'
    if (!inviteDate.trim()) nextErrors.date = 'Pick a date.'
    if (!inviteTime.trim()) nextErrors.time = 'Pick a time.'
    setInviteErrors(nextErrors)
    if (Object.keys(nextErrors).length > 0) return

    setIsSendingInvite(true)
    try {
      await onInviteToEat({
        to_username: inviteTarget.username,
        to_email: inviteTarget.email || '',
        restaurant_name: inviteRestaurant,
        date: inviteDate,
        time: inviteTime,
        message: inviteMessage,
      })
      setInviteOpen(false)
      setInviteSuccessMessage(`Invite sent to ${inviteTarget.username}.`)
      window.setTimeout(() => setInviteSuccessMessage(''), 2800)
    } finally {
      setIsSendingInvite(false)
    }
  }

  return (
    <section className="rounded-3xl bg-card border border-border/50 p-6 md:p-8 space-y-6">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-xl bg-primary/10 flex items-center justify-center">
            <Users className="w-5 h-5 text-primary" />
          </div>
          <div>
            <h3 className="text-lg font-semibold">Taste Matches</h3>
            <p className="text-sm text-muted-foreground">People and places aligned with your palate</p>
          </div>
        </div>
        {inviteSuccessMessage ? (
          <div className="inline-flex items-center gap-1.5 rounded-full bg-primary/10 text-primary px-3 py-1 text-xs font-medium">
            <CheckCircle2 className="w-3.5 h-3.5" />
            {inviteSuccessMessage}
          </div>
        ) : null}
      </div>
      <Tabs
        defaultValue="taste"
        className="space-y-4"
        onValueChange={(v) => setActiveTab(v === 'restaurant' ? 'restaurant' : 'taste')}
      >
        <TabsList className="grid w-full grid-cols-2">
          <TabsTrigger value="taste">Taste Matches</TabsTrigger>
          <TabsTrigger value="restaurant">Restaurant Matches</TabsTrigger>
        </TabsList>
        <TabsContent value="taste" className="space-y-3">
          {users.length === 0 ? (
            <div className="text-center py-8">
              <div className="w-16 h-16 mx-auto mb-4 rounded-full bg-primary/10 flex items-center justify-center">
                <Heart className="w-8 h-8 text-primary/50" />
              </div>
              <p className="text-muted-foreground mb-2">No taste matches yet</p>
              <p className="text-sm text-muted-foreground/70">
                Upload more dishes to find users with similar taste profiles.
              </p>
            </div>
          ) : (
            users.map((user) => (
              <UserCard
                key={user.username}
                user={user}
                onInvite={() => openInviteModal(user)}
                onEmail={() => window.location.href = buildMailto(user)}
              />
            ))
          )}
        </TabsContent>
        <TabsContent value="restaurant" className="space-y-4">
          <div className="rounded-2xl border border-border/40 bg-secondary/30 p-4 space-y-3">
            <p className="text-sm text-muted-foreground">
              Enter your ZIP to rank practical restaurant matches for your profile.
            </p>
            <Input
              value={zip}
              onChange={(e) => setZip(e.target.value.replace(/[^\d]/g, '').slice(0, 5))}
              placeholder="ZIP code (e.g. 60614)"
              className="max-w-xs"
            />
          </div>
          <div className="space-y-3">
            {zip.length !== 5 ? (
              <p className="text-sm text-muted-foreground">Enter a 5-digit ZIP to see ranked results.</p>
            ) : isLoadingRestaurants ? (
              <p className="text-sm text-muted-foreground">Ranking restaurants...</p>
            ) : restaurantError ? (
              <p className="text-sm text-destructive">{restaurantError}</p>
            ) : restaurants.length === 0 ? (
              <p className="text-sm text-muted-foreground">No matches found for this ZIP yet.</p>
            ) : (
              restaurants.map((r) => (
                <div key={r.id} className="p-4 rounded-2xl bg-secondary/40 border border-border/30">
                  <div className="flex items-start justify-between gap-3">
                    <div>
                      <h4 className="font-medium">{r.name}</h4>
                      <p className="text-xs text-muted-foreground">
                        {(r.cuisine_tags || []).slice(0, 2).join(' · ') || 'Mixed cuisine'} · {r.zip_code}
                      </p>
                    </div>
                    <div className="text-right">
                      <Sparkles className="w-4 h-4 text-primary ml-auto" />
                      <p className="text-xs font-semibold text-primary mt-1">{Math.round(r.compatibility_score)}%</p>
                    </div>
                  </div>
                  <p className="text-sm text-muted-foreground mt-2">{r.explanation}</p>
                  <p className="text-[11px] text-muted-foreground/80 mt-2">
                    Cuisine {Math.round(r.score_breakdown.cuisine_match.score * 100)} · Dish {Math.round(r.score_breakdown.dish_match.score * 100)} ·
                    Traits {Math.round(r.score_breakdown.trait_match.score * 100)} · Location {Math.round(r.score_breakdown.location_match.score * 100)}
                  </p>
                  <div className="mt-3">
                    <ActionButton recommendation={r} />
                  </div>
                  <div className="mt-2">
                    <Button
                      variant="ghost"
                      size="sm"
                      className="h-7 rounded-full text-xs text-muted-foreground"
                      onClick={() => {
                        setInviteRestaurant(r.name)
                        setInviteContextRestaurant(r.name)
                      }}
                    >
                      Use for invite
                    </Button>
                  </div>
                </div>
              ))
            )}
          </div>
        </TabsContent>
      </Tabs>
      <Dialog open={inviteOpen} onOpenChange={setInviteOpen}>
        <DialogContent className="rounded-2xl sm:max-w-md border-border/60 p-0 overflow-hidden">
          <DialogHeader className="px-6 pt-6 pb-4 border-b border-border/50">
            <DialogTitle>
              {inviteTarget
                ? `Invite ${inviteTarget.username} to ${inviteRestaurant || 'eat'}`
                : 'Invite to eat'}
            </DialogTitle>
            <DialogDescription>
              Send a simple plan with restaurant and time details.
            </DialogDescription>
          </DialogHeader>
          <div className="px-6 py-4 space-y-4">
            {inviteContextRestaurant ? (
              <div className="rounded-xl bg-secondary/40 border border-border/40 px-3 py-2">
                <p className="text-[11px] uppercase tracking-wide text-muted-foreground">Selected from restaurant matches</p>
                <p className="text-sm font-medium">{inviteContextRestaurant}</p>
              </div>
            ) : null}
            <div className="space-y-1.5">
              <label className="text-xs text-muted-foreground">Restaurant *</label>
              <Input
                placeholder="Restaurant"
                value={inviteRestaurant}
                onChange={(e) => {
                  setInviteRestaurant(e.target.value)
                  if (inviteErrors.restaurant) setInviteErrors((x) => ({ ...x, restaurant: undefined }))
                }}
                className={inviteErrors.restaurant ? 'border-destructive/60' : ''}
              />
              {inviteErrors.restaurant ? <p className="text-xs text-destructive">{inviteErrors.restaurant}</p> : null}
            </div>
            <div className="grid grid-cols-2 gap-3">
              <div className="space-y-1.5">
                <label className="text-xs text-muted-foreground">Date *</label>
                <Input
                  type="date"
                  value={inviteDate}
                  onChange={(e) => {
                    setInviteDate(e.target.value)
                    if (inviteErrors.date) setInviteErrors((x) => ({ ...x, date: undefined }))
                  }}
                  className={inviteErrors.date ? 'border-destructive/60' : ''}
                />
                {inviteErrors.date ? <p className="text-xs text-destructive">{inviteErrors.date}</p> : null}
              </div>
              <div className="space-y-1.5">
                <label className="text-xs text-muted-foreground">Time *</label>
                <Input
                  type="time"
                  value={inviteTime}
                  onChange={(e) => {
                    setInviteTime(e.target.value)
                    if (inviteErrors.time) setInviteErrors((x) => ({ ...x, time: undefined }))
                  }}
                  className={inviteErrors.time ? 'border-destructive/60' : ''}
                />
                {inviteErrors.time ? <p className="text-xs text-destructive">{inviteErrors.time}</p> : null}
              </div>
            </div>
            <div className="space-y-1.5">
              <label className="text-xs text-muted-foreground">Optional note</label>
              <Textarea
                placeholder="Optional note"
                value={inviteMessage}
                onChange={(e) => setInviteMessage(e.target.value)}
                rows={3}
              />
            </div>
          </div>
          <DialogFooter className="px-6 py-4 border-t border-border/50 bg-secondary/20">
            <Button
              variant="outline"
              onClick={() => {
                if (!inviteTarget) return
                window.location.href = buildMailto(inviteTarget, true)
              }}
            >
              Send email instead
            </Button>
            <Button onClick={handleSendInvite} disabled={!inviteTarget || isSendingInvite} className="min-w-28">
              {isSendingInvite ? 'Sending...' : 'Send invite'}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </section>
  )
}

function ActionButton({ recommendation }: { recommendation: RestaurantRecommendation }) {
  const action = recommendation.booking_action
  if (!action || action.type === 'none' || !action.url) {
    return (
      <Button variant="ghost" size="sm" className="text-muted-foreground" disabled>
        No action available
      </Button>
    )
  }
  const isExternal = action.type === 'resy' || action.type === 'opentable' || action.type === 'website'
  return (
    <Button
      variant="outline"
      size="sm"
      className="rounded-full"
      onClick={() => {
        if (isExternal) window.open(action.url, '_blank', 'noopener,noreferrer')
        else window.location.href = action.url
      }}
    >
      {action.label}
    </Button>
  )
}

function UserCard({
  user,
  onInvite,
  onEmail,
}: {
  user: CompatibleUser
  onInvite: () => void
  onEmail: () => void
}) {
  const getMatchColor = (score: number) => {
    if (score >= 90) return 'text-primary'
    if (score >= 80) return 'text-chart-2'
    return 'text-muted-foreground'
  }

  return (
    <div className="p-4 rounded-2xl bg-secondary/40 border border-border/30 hover:border-primary/30 transition-colors">
      <div className="flex items-start gap-4">
        {/* Avatar */}
        <div className="w-12 h-12 rounded-xl bg-primary/10 flex items-center justify-center flex-shrink-0">
          <span className="text-lg font-semibold text-primary">
            {user.username.charAt(0).toUpperCase()}
          </span>
        </div>

        {/* Content */}
        <div className="flex-1 min-w-0">
          <div className="flex items-center justify-between gap-2 mb-1">
            <h4 className="font-medium truncate">@{user.username}</h4>
            <div className="flex items-center gap-1">
              <Sparkles className={`w-4 h-4 ${getMatchColor(user.compatibility_score)}`} />
              <span className={`text-sm font-semibold ${getMatchColor(user.compatibility_score)}`}>
                {user.compatibility_score}%
              </span>
            </div>
          </div>
          <p className="text-sm text-muted-foreground line-clamp-2">
            {user.reason}
          </p>
          <div className="mt-3 flex items-center gap-2">
            <Button size="sm" className="rounded-full" onClick={onInvite}>
              Invite to eat
            </Button>
            <Button size="sm" variant="outline" className="rounded-full" onClick={onEmail}>
              Email
            </Button>
          </div>
        </div>
      </div>
    </div>
  )
}
