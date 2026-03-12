'use client'

import { useMemo } from 'react'
import { Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import { TrendingUp } from 'lucide-react'
import type { TasteTrendPoint } from '@/lib/types'

interface TasteTrendsProps {
  history: TasteTrendPoint[]
}

const SERIES = [
  { key: 'sweet', label: 'Sweet', color: '#ec4899' },
  { key: 'spicy', label: 'Spicy', color: '#ef4444' },
  { key: 'rich', label: 'Comfort Food', color: '#f59e0b' },
  { key: 'fresh', label: 'Fresh', color: '#10b981' },
  { key: 'dessert', label: 'Dessert', color: '#8b5cf6' },
] as const

export function TasteTrends({ history }: TasteTrendsProps) {
  const seriesNameByKey = useMemo(
    () =>
      Object.fromEntries(
        SERIES.map((s) => [s.key, s.label])
      ) as Record<string, string>,
    []
  )

  const data = useMemo(() => {
    return history.map((h, i) => ({
      step: Math.max(1, Number(h.interaction_count || i + 1)),
      sweet: Number(h.dimensions.sweet_leaning || 0),
      spicy: Number(h.dimensions.spicy_leaning || 0),
      rich: Number(h.dimensions.richness_preference || 0),
      fresh: Number(h.dimensions.freshness_preference || 0),
      dessert: Number(h.dimensions.dessert_affinity || 0),
    }))
  }, [history])
  const interactionTicks = useMemo(
    () => Array.from(new Set(data.map((d) => d.step))).sort((a, b) => a - b),
    [data]
  )

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
          <p className="text-sm text-muted-foreground">Taste traits over profile interactions</p>
        </div>
      </div>
      <div className="flex flex-wrap gap-3 text-xs">
        {SERIES.map((series) => (
          <div key={series.key} className="inline-flex items-center gap-2 rounded-full border border-border/60 bg-background/70 px-3 py-1">
            <span
              className="inline-block h-2.5 w-2.5 rounded-full"
              style={{ backgroundColor: series.color }}
              aria-hidden="true"
            />
            <span className="text-muted-foreground">{series.label}</span>
          </div>
        ))}
      </div>
      <div className="h-[260px]">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={data} margin={{ top: 8, right: 16, left: 18, bottom: 24 }}>
            <XAxis
              type="number"
              dataKey="step"
              domain={['dataMin', 'dataMax']}
              ticks={interactionTicks}
              allowDecimals={false}
              tickLine={false}
              axisLine={false}
              tickMargin={8}
              label={{ value: 'Profile Interactions', position: 'insideBottom', offset: -10 }}
            />
            <YAxis
              domain={[0, 100]}
              tickLine={false}
              axisLine={false}
              tickMargin={10}
              width={72}
              tickFormatter={(value: number) => `${Math.round(value)}%`}
              label={{ value: 'Taste Signal Strength', angle: -90, position: 'left', offset: 2 }}
            />
            <Tooltip
              formatter={(value: number, name: string) => {
                const label = seriesNameByKey[name] || name
                return [`${Math.round(Number(value || 0))}%`, label]
              }}
              labelFormatter={(value) => `Interaction ${value}`}
              contentStyle={{ borderRadius: 12, borderColor: 'hsl(var(--border))' }}
            />
            <Line type="monotone" dataKey="sweet" stroke="#ec4899" dot={false} strokeWidth={2} name="Sweet" />
            <Line type="monotone" dataKey="spicy" stroke="#ef4444" dot={false} strokeWidth={2} name="Spicy" />
            <Line type="monotone" dataKey="rich" stroke="#f59e0b" dot={false} strokeWidth={2} name="Comfort Food" />
            <Line type="monotone" dataKey="fresh" stroke="#10b981" dot={false} strokeWidth={2} name="Fresh" />
            <Line type="monotone" dataKey="dessert" stroke="#8b5cf6" dot={false} strokeWidth={2} name="Dessert" />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </section>
  )
}
