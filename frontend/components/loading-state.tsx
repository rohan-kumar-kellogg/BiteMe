'use client'

import { BiteMePieIcon } from '@/components/bite-me-logo'

export function LoadingState() {
  return (
    <div className="min-h-screen flex items-center justify-center px-4 bg-gradient-to-b from-primary/[0.03] via-transparent to-primary/[0.02]">
      <div className="text-center space-y-8">
        <div className="relative inline-flex items-center justify-center">
          {/* Outer ring animations */}
          <div className="absolute w-36 h-36 rounded-full border-2 border-primary/10 animate-ping" />
          <div className="absolute w-32 h-32 rounded-full border-2 border-primary/20 animate-pulse" />
          
          {/* Pie Icon */}
          <div className="relative animate-pulse">
            <BiteMePieIcon size={80} />
          </div>
        </div>
        
        <div className="space-y-3">
          <h2 className="text-xl md:text-2xl font-semibold text-foreground">
            Analyzing your taste...
          </h2>
          <p className="text-sm md:text-base text-muted-foreground">
            Preparing your flavor fingerprint
          </p>
        </div>

        {/* Loading dots */}
        <div className="flex items-center justify-center gap-2">
          <span className="w-2.5 h-2.5 rounded-full bg-primary animate-bounce" style={{ animationDelay: '0ms' }} />
          <span className="w-2.5 h-2.5 rounded-full bg-primary animate-bounce" style={{ animationDelay: '150ms' }} />
          <span className="w-2.5 h-2.5 rounded-full bg-primary animate-bounce" style={{ animationDelay: '300ms' }} />
        </div>
      </div>
    </div>
  )
}
