'use client'

import { useEffect, useMemo, useState } from 'react'
import {
  Radar,
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  ResponsiveContainer,
  Tooltip,
} from 'recharts'
import { ChevronDown, ChevronUp } from 'lucide-react'
import type { TasteDimension } from '@/lib/types'

interface FlavorFingerprintProps {
  dimensions: TasteDimension[]
  refreshToken?: number
}

export function FlavorFingerprint({ dimensions, refreshToken = 0 }: FlavorFingerprintProps) {
  const [showAllDimensions, setShowAllDimensions] = useState(false)
  const [showUpdateNotice, setShowUpdateNotice] = useState(false)
  const rankedDimensions = useMemo(() => {
    return [...dimensions].sort((a, b) => b.value - a.value)
  }, [dimensions])

  // Keep chart clean: only strongest 6 dimensions.
  const radarData = useMemo(() => {
    return rankedDimensions.slice(0, 6).map((d) => ({
      label: d.label,
      key: d.key,
      value: d.value,
      description: d.description || '',
      fullMark: 100,
    }))
  }, [rankedDimensions])

  // Get top traits for display
  const topTraits = useMemo(() => {
    return rankedDimensions.slice(0, 4)
  }, [rankedDimensions])

  const dominantTrait = rankedDimensions[0]

  useEffect(() => {
    if (!refreshToken) return
    setShowUpdateNotice(true)
    const t = window.setTimeout(() => setShowUpdateNotice(false), 1400)
    return () => window.clearTimeout(t)
  }, [refreshToken])

  if (dimensions.length === 0) {
    return (
      <section className="rounded-3xl bg-card border border-border/50 p-6 md:p-8">
        <h3 className="text-lg font-semibold mb-4">Your Flavor Fingerprint</h3>
        <div className="text-center py-12 text-muted-foreground">
          <p>Upload more dishes to reveal your unique flavor profile</p>
        </div>
      </section>
    )
  }

  return (
    <section className="rounded-3xl bg-card border border-border/50 overflow-hidden">
      <div className="p-6 md:p-8 space-y-6">
        <div className="flex items-center justify-between">
          <div>
            <h3 className="text-lg font-semibold">Flavor Fingerprint</h3>
            <p className="text-sm text-muted-foreground">Your unique taste profile visualized</p>
          </div>
        </div>
        {dominantTrait && (
          <div className="flex items-center gap-2">
            <div className="inline-flex items-center rounded-full bg-primary/10 border border-primary/20 px-3 py-1">
              <span className="text-xs text-muted-foreground mr-1.5">Dominant trait:</span>
              <span className="text-xs font-semibold text-primary">{dominantTrait.label}</span>
            </div>
            {showUpdateNotice ? (
              <span className="text-xs text-primary/90">Taste profile updated</span>
            ) : null}
          </div>
        )}

        {/* Radar Chart */}
        <div className="h-[280px] md:h-[320px] -mx-4">
          <ResponsiveContainer width="100%" height="100%">
            <RadarChart cx="50%" cy="50%" outerRadius="70%" data={radarData}>
              <defs>
                <filter id="fingerprintGlow" x="-40%" y="-40%" width="180%" height="180%">
                  <feGaussianBlur stdDeviation="2.5" result="blur" />
                  <feMerge>
                    <feMergeNode in="blur" />
                    <feMergeNode in="SourceGraphic" />
                  </feMerge>
                </filter>
              </defs>
              <PolarGrid stroke="var(--border)" strokeOpacity={0.5} />
              <PolarAngleAxis 
                dataKey="label" 
                tick={{ fontSize: 11, fill: 'var(--muted-foreground)' }}
                tickLine={false}
              />
              <PolarRadiusAxis 
                angle={30} 
                domain={[0, 100]} 
                tick={false}
                axisLine={false}
              />
              <Tooltip
                formatter={(value: number) => [`${value}%`, 'Score']}
                labelFormatter={(_, payload) => {
                  const p = payload?.[0]?.payload as { label?: string; description?: string } | undefined
                  if (!p) return ''
                  return p.label || ''
                }}
                contentStyle={{
                  borderRadius: 12,
                  border: '1px solid hsl(var(--border))',
                  background: 'hsl(var(--card))',
                  fontSize: 12,
                  color: 'hsl(var(--foreground))',
                }}
                cursor={false}
              />
              <Radar
                name="Taste Profile"
                dataKey="value"
                stroke="var(--primary)"
                fill="var(--primary)"
                fillOpacity={0.3}
                strokeWidth={2.6}
                filter="url(#fingerprintGlow)"
                dot={{ r: 3.5, fill: 'var(--primary)', stroke: 'var(--background)', strokeWidth: 1 }}
                activeDot={{ r: 5 }}
                isAnimationActive
                animationDuration={700}
                animationEasing="ease-out"
              />
            </RadarChart>
          </ResponsiveContainer>
        </div>

        {/* Top Traits */}
        <div className="grid grid-cols-2 gap-3">
          {topTraits.map((trait) => (
            <div 
              key={trait.key} 
              className="p-4 rounded-2xl bg-secondary/50 border border-border/30"
            >
              <div className="flex items-center justify-between mb-2">
                <span className="text-sm font-medium">{trait.label}</span>
                <span className="text-sm font-semibold text-primary">{trait.value}%</span>
              </div>
              <div className="h-1.5 bg-muted rounded-full overflow-hidden">
                <div 
                  className="h-full bg-primary rounded-full transition-all duration-500"
                  style={{ width: `${trait.value}%` }}
                />
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* All Dimensions */}
      <div className="border-t border-border/50 p-6 md:p-8 bg-secondary/20">
        <button
          type="button"
          onClick={() => setShowAllDimensions((v) => !v)}
          className="w-full flex items-center justify-between text-left"
        >
          <h4 className="text-sm font-medium text-muted-foreground">All Taste Dimensions</h4>
          {showAllDimensions ? (
            <span className="inline-flex items-center gap-1 text-xs text-muted-foreground">
              Hide <ChevronUp className="w-4 h-4" />
            </span>
          ) : (
            <span className="inline-flex items-center gap-1 text-xs text-muted-foreground">
              Show <ChevronDown className="w-4 h-4" />
            </span>
          )}
        </button>
        {showAllDimensions && (
          <div className="space-y-3 mt-4">
            {dimensions.map((dim) => (
              <div key={dim.key} className="flex items-center gap-4">
                <span className="text-sm w-20 text-muted-foreground">{dim.label}</span>
                <div className="flex-1 h-2 bg-muted rounded-full overflow-hidden">
                  <div 
                    className="h-full bg-primary/70 rounded-full transition-all duration-500"
                    style={{ width: `${dim.value}%` }}
                  />
                </div>
                <span className="text-sm font-medium w-10 text-right">{dim.value}</span>
              </div>
            ))}
          </div>
        )}
      </div>
    </section>
  )
}
