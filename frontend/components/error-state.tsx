'use client'

import { AlertCircle, RefreshCw, Home } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { BiteMeLogo } from '@/components/bite-me-logo'

interface ErrorStateProps {
  message?: string
  onRetry?: () => void
  onGoHome?: () => void
}

export function ErrorState({ 
  message = 'Something went wrong. Please try again.', 
  onRetry, 
  onGoHome 
}: ErrorStateProps) {
  return (
    <div className="min-h-screen flex items-center justify-center px-4">
      <div className="text-center space-y-6 max-w-md">
        <div className="flex items-center justify-center mb-2 opacity-50">
          <BiteMeLogo iconSize={24} />
        </div>
        
        <div className="inline-flex items-center justify-center w-20 h-20 rounded-3xl bg-destructive/10">
          <AlertCircle className="w-10 h-10 text-destructive" strokeWidth={1.5} />
        </div>
        
        <div className="space-y-2">
          <h2 className="text-xl font-semibold">Oops!</h2>
          <p className="text-muted-foreground">
            {message}
          </p>
        </div>

        <div className="flex flex-col sm:flex-row items-center justify-center gap-3">
          {onRetry && (
            <Button onClick={onRetry} variant="default" className="rounded-full gap-2">
              <RefreshCw className="w-4 h-4" />
              Try Again
            </Button>
          )}
          {onGoHome && (
            <Button onClick={onGoHome} variant="outline" className="rounded-full gap-2">
              <Home className="w-4 h-4" />
              Start Over
            </Button>
          )}
        </div>
      </div>
    </div>
  )
}
