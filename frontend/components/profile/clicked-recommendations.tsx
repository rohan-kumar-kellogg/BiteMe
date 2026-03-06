'use client'

import { Heart, Sparkles } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import type { ClickedRecommendation } from '@/lib/types'

interface ClickedRecommendationsProps {
  items: ClickedRecommendation[]
  onRemove: (eventId: string) => Promise<void>
}

export function ClickedRecommendations({ items, onRemove }: ClickedRecommendationsProps) {
  return (
    <section className="rounded-3xl bg-card border border-border/50 p-6 md:p-8 space-y-5">
      <div className="flex items-center gap-3">
        <div className="w-10 h-10 rounded-xl bg-primary/10 flex items-center justify-center">
          <Heart className="w-5 h-5 text-primary" />
        </div>
        <div>
          <h3 className="text-lg font-semibold">Clicked Picks</h3>
          <p className="text-sm text-muted-foreground">Interest signals from recommendation clicks</p>
        </div>
      </div>
      {items.length === 0 ? (
        <div className="text-sm text-muted-foreground">
          Click a few suggestions in "You May Also Like" and they will show up here.
        </div>
      ) : (
        <div className="flex flex-wrap gap-2">
          {items.map((item) => (
            <Badge
              key={item.event_id}
              variant="secondary"
              className="rounded-full px-3 py-1 h-8 text-xs gap-2 border border-border/30 bg-secondary/40"
            >
              <Sparkles className="w-3 h-3" />
              <span>{item.dish_label}</span>
              <Button
                variant="ghost"
                size="sm"
                className="h-5 px-1 text-[10px] -mr-1"
                onClick={() => onRemove(item.event_id)}
              >
                Remove
              </Button>
            </Badge>
          ))}
        </div>
      )}
    </section>
  )
}
