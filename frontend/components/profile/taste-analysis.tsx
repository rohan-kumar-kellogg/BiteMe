'use client'

import { Lightbulb, TrendingUp } from 'lucide-react'
import type { UserProfile } from '@/lib/types'

interface TasteAnalysisProps {
  profile: UserProfile
}

export function TasteAnalysis({ profile }: TasteAnalysisProps) {
  const { taste_profile, observations } = profile
  const hasUploads = profile.total_uploads > 0

  if (!hasUploads) {
    return (
      <section className="rounded-3xl bg-card border border-border/50 p-6 md:p-8 space-y-4">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-xl bg-primary/10 flex items-center justify-center">
            <Lightbulb className="w-5 h-5 text-primary" />
          </div>
          <div>
            <h3 className="text-lg font-semibold">Taste Analysis</h3>
            <p className="text-sm text-muted-foreground">Learning starts after your first uploads</p>
          </div>
        </div>
        <p className="text-sm text-muted-foreground">
          BiteMe needs a few dish photos before it can generate a reliable personality read. Add your first uploads and
          this section will switch from onboarding to personalized insights.
        </p>
      </section>
    )
  }

  if (!taste_profile.analysis && observations.length === 0) {
    return null
  }

  return (
    <section className="rounded-3xl bg-card border border-border/50 p-6 md:p-8 space-y-6">
      <div className="flex items-center gap-3">
        <div className="w-10 h-10 rounded-xl bg-primary/10 flex items-center justify-center">
          <Lightbulb className="w-5 h-5 text-primary" />
        </div>
        <div>
          <h3 className="text-lg font-semibold">Taste Analysis</h3>
          <p className="text-sm text-muted-foreground">What your food choices reveal</p>
        </div>
      </div>

      {/* Main Analysis */}
      {taste_profile.analysis && (
        <div className="p-5 rounded-2xl bg-secondary/40 border border-border/30">
          <p className="text-foreground/90 leading-relaxed text-pretty">
            {taste_profile.analysis}
          </p>
        </div>
      )}

      {/* Observations */}
      {observations.length > 0 && (
        <div className="space-y-3">
          <div className="flex items-center gap-2 text-sm font-medium text-muted-foreground">
            <TrendingUp className="w-4 h-4" />
            <span>Key Observations</span>
          </div>
          <ul className="space-y-3">
            {observations.map((observation, index) => (
              <li 
                key={index} 
                className="flex gap-3 text-sm text-foreground/80 leading-relaxed"
              >
                <span className="mt-1.5 w-1.5 h-1.5 rounded-full bg-primary flex-shrink-0" />
                <span>{observation}</span>
              </li>
            ))}
          </ul>
        </div>
      )}

    </section>
  )
}
