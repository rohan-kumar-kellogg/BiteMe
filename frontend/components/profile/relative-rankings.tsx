'use client'

import { TrendingUp, TrendingDown, Minus, BarChart3 } from 'lucide-react'
import type { RelativeRanking } from '@/lib/types'

interface RelativeRankingsProps {
  rankings: RelativeRanking[]
}

export function RelativeRankings({ rankings }: RelativeRankingsProps) {
  if (rankings.length === 0) {
    return null
  }

  return (
    <section className="rounded-3xl bg-card border border-border/50 p-6 md:p-8 space-y-6">
      <div className="flex items-center gap-3">
        <div className="w-10 h-10 rounded-xl bg-primary/10 flex items-center justify-center">
          <BarChart3 className="w-5 h-5 text-primary" />
        </div>
        <div>
          <h3 className="text-lg font-semibold">How You Compare</h3>
          <p className="text-sm text-muted-foreground">Your rankings vs. other food lovers</p>
        </div>
      </div>

      <div className="grid gap-3 sm:grid-cols-2">
        {rankings.map((ranking, index) => (
          <RankingCard key={index} ranking={ranking} />
        ))}
      </div>
    </section>
  )
}

function RankingCard({ ranking }: { ranking: RelativeRanking }) {
  const getIcon = () => {
    switch (ranking.direction) {
      case 'higher':
        return <TrendingUp className="w-4 h-4 text-primary" />
      case 'lower':
        return <TrendingDown className="w-4 h-4 text-muted-foreground" />
      default:
        return <Minus className="w-4 h-4 text-muted-foreground" />
    }
  }

  const getColor = () => {
    if (ranking.percentile >= 75) return 'bg-primary/10 border-primary/20'
    if (ranking.percentile >= 50) return 'bg-secondary border-border/50'
    return 'bg-muted/50 border-border/30'
  }

  return (
    <div className={`p-4 rounded-2xl border ${getColor()} transition-colors`}>
      <div className="flex items-start justify-between gap-3">
        <div className="space-y-1">
          <div className="flex items-center gap-2">
            {getIcon()}
            <span className="text-sm font-medium">{ranking.label}</span>
          </div>
          <p className="text-xs text-muted-foreground">{ranking.description}</p>
        </div>
        {ranking.percentile > 0 && ranking.direction !== 'similar' && (
          <div className="text-right">
            <span className="text-lg font-semibold text-primary">{ranking.percentile}%</span>
          </div>
        )}
      </div>
      
      {/* Progress indicator */}
      <div className="mt-3 h-1 bg-muted rounded-full overflow-hidden">
        <div 
          className="h-full bg-primary/60 rounded-full transition-all duration-500"
          style={{ width: `${ranking.percentile}%` }}
        />
      </div>
    </div>
  )
}
