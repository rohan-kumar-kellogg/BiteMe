'use client'

import { useMemo } from 'react'
import { Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import { TrendingUp } from 'lucide-react'
import type { TasteTrendPoint } from '@/lib/types'

interface TasteTrendsProps {
  history: TasteTrendPoint[]
}

export function TasteTrends({ history }: TasteTrendsProps) {
  const data = useMemo(() => {
    return history.map((h, i) => ({
      step: Math.max(1, Number(h.upload_count || i + 1)),
      sweet: Number(h.dimensions.sweet_leaning || 0),
      spicy: Number(h.dimensions.spicy_leaning || 0),
      rich: Number(h.dimensions.richness_preference || 0),
      fresh: Number(h.dimensions.freshness_preference || 0),
      dessert: Number(h.dimensions.dessert_affinity || 0),
    }))
  }, [history])

  if (data.length < 2) {
    return (
      <section className="rounded-3xl bg-card border border-border/50 p-6 md:p-8 space-y-4">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-xl bg-primary/10 flex items-center justify-center">
            <TrendingUp className="w-5 h-5 text-primary" />
          </div>
          <div>
            <h3 className="text-lg font-semibold">Taste Trends</h3>
            <p className="text-sm text-muted-foreground">Track how your palate shifts over uploads</p>
          </div>
        </div>
        <p className="text-sm text-muted-foreground">
          Upload a couple more dishes to start seeing trend lines.
        </p>
      </section>
    )
  }

  return (
    <section className="rounded-3xl bg-card border border-border/50 p-6 md:p-8 space-y-4">
      <div className="flex items-center gap-3">
        <div className="w-10 h-10 rounded-xl bg-primary/10 flex items-center justify-center">
          <TrendingUp className="w-5 h-5 text-primary" />
        </div>
        <div>
          <h3 className="text-lg font-semibold">Taste Trends</h3>
          <p className="text-sm text-muted-foreground">How your key traits evolve over time</p>
        </div>
      </div>
      <div className="h-[260px] -mx-2">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={data}>
            <XAxis dataKey="step" tickLine={false} axisLine={false} />
            <YAxis domain={[0, 100]} tickLine={false} axisLine={false} width={30} />
            <Tooltip />
            <Line type="monotone" dataKey="sweet" stroke="#ec4899" dot={false} strokeWidth={2} name="Sweet" />
            <Line type="monotone" dataKey="spicy" stroke="#ef4444" dot={false} strokeWidth={2} name="Spicy" />
            <Line type="monotone" dataKey="rich" stroke="#f59e0b" dot={false} strokeWidth={2} name="Richness" />
            <Line type="monotone" dataKey="fresh" stroke="#10b981" dot={false} strokeWidth={2} name="Freshness" />
            <Line type="monotone" dataKey="dessert" stroke="#8b5cf6" dot={false} strokeWidth={2} name="Dessert" />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </section>
  )
}
