'use client'

import { Sparkles, Camera, TrendingUp, Users, ArrowLeft } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { UploadModule } from '@/components/profile/upload-module'
import { BiteMeLogo } from '@/components/bite-me-logo'

interface NewUserStateProps {
  username: string
  onUpload: (file: File) => Promise<void>
  isUploading: boolean
  onBack?: () => void
}

export function NewUserState({ username, onUpload, isUploading, onBack }: NewUserStateProps) {
  return (
    <div className="min-h-screen flex flex-col">
      {/* Header */}
      <header className="sticky top-0 z-50 bg-background/80 backdrop-blur-lg border-b border-border/50">
        <div className="max-w-3xl mx-auto px-4 h-14 flex items-center justify-between">
          {onBack && (
            <Button 
              variant="ghost" 
              size="sm" 
              onClick={onBack}
              className="gap-2 -ml-2"
            >
              <ArrowLeft className="w-4 h-4" />
              <span className="hidden sm:inline">Back</span>
            </Button>
          )}
          {!onBack && <div />}
          
          <BiteMeLogo iconSize={28} />
          
          <div className="w-16" />
        </div>
      </header>

      <div className="flex-1 px-4 py-8 md:py-12">
        <div className="max-w-2xl mx-auto space-y-8">
          {/* Welcome Header */}
          <div className="text-center space-y-4">
            <h1 className="text-3xl md:text-4xl font-semibold tracking-tight">
              Welcome, @{username}!
            </h1>
            <p className="text-lg text-muted-foreground text-pretty max-w-md mx-auto">
              Let's build your unique flavor profile. Start by uploading photos of your favorite dishes.
            </p>
          </div>

        {/* Upload Module - Prominent */}
        <UploadModule onUpload={onUpload} isUploading={isUploading} />

        {/* How It Works */}
        <div className="rounded-3xl bg-card border border-border/50 p-6 md:p-8 space-y-6">
          <h2 className="text-lg font-semibold text-center">How It Works</h2>
          
          <div className="grid gap-4 sm:grid-cols-3">
            <StepCard 
              number={1}
              icon={Camera}
              title="Upload Food Photos"
              description="Share pictures of dishes you love eating"
            />
            <StepCard 
              number={2}
              icon={Sparkles}
              title="Build Your Profile"
              description="We analyze your taste preferences over time"
            />
            <StepCard 
              number={3}
              icon={TrendingUp}
              title="Discover Insights"
              description="Learn what makes your palate unique"
            />
          </div>
        </div>

        {/* Benefits Preview */}
        <div className="rounded-3xl bg-secondary/30 border border-border/30 p-6 space-y-4">
          <h3 className="text-sm font-medium text-muted-foreground text-center">
            What you'll unlock
          </h3>
          <div className="flex flex-wrap justify-center gap-2">
            <BenefitPill icon={Sparkles} label="Taste Archetype" />
            <BenefitPill icon={TrendingUp} label="Flavor Fingerprint" />
            <BenefitPill icon={Users} label="Compatible Matches" />
          </div>
        </div>
        </div>
      </div>

      {/* Footer */}
      <footer className="py-8 text-center">
        <p className="text-xs text-muted-foreground/60 italic">
          A little too into what your food says about you.
        </p>
      </footer>
    </div>
  )
}

function StepCard({ 
  number, 
  icon: Icon, 
  title, 
  description 
}: { 
  number: number
  icon: React.ElementType
  title: string
  description: string 
}) {
  return (
    <div className="text-center p-4 rounded-2xl bg-secondary/40 border border-border/30">
      <div className="relative inline-flex items-center justify-center w-12 h-12 rounded-xl bg-primary/10 mb-3">
        <Icon className="w-6 h-6 text-primary" />
        <span className="absolute -top-1 -right-1 w-5 h-5 rounded-full bg-primary text-primary-foreground text-xs font-semibold flex items-center justify-center">
          {number}
        </span>
      </div>
      <h3 className="text-sm font-medium mb-1">{title}</h3>
      <p className="text-xs text-muted-foreground">{description}</p>
    </div>
  )
}

function BenefitPill({ icon: Icon, label }: { icon: React.ElementType; label: string }) {
  return (
    <div className="flex items-center gap-2 px-4 py-2 rounded-full bg-card border border-border/50">
      <Icon className="w-4 h-4 text-primary" />
      <span className="text-sm font-medium">{label}</span>
    </div>
  )
}
