'use client'

import Image from 'next/image'

// Pie icon using the exact uploaded SVG
interface BiteMePieIconProps {
  className?: string
  size?: number
}

export function BiteMePieIcon({ className = '', size = 32 }: BiteMePieIconProps) {
  // Scale the image slightly larger and clip overflow to eliminate white edges
  const imageSize = Math.round(size * 1.15)
  return (
    <div 
      className={`bg-background overflow-hidden flex items-center justify-center ${className}`} 
      style={{ width: size, height: size }}
    >
      <Image
        src="/images/bite-me-pie.svg"
        alt="Bite Me"
        width={imageSize}
        height={imageSize}
        className="mix-blend-darken"
        priority
      />
    </div>
  )
}

// Full logo with pie icon on left, "BITE ME" wordmark on right
interface BiteMeLogoProps {
  className?: string
  iconSize?: number
}

export function BiteMeLogo({ className = '', iconSize = 50 }: BiteMeLogoProps) {
  return (
    <div className={`flex items-center gap-3 ${className}`}>
      <BiteMePieIcon size={iconSize} />
      <span className="font-bold text-primary tracking-widest text-lg">BITE ME</span>
    </div>
  )
}

// Large pie icon for hero/background with low opacity
interface BiteMePieBackgroundProps {
  className?: string
  size?: number
}

export function BiteMePieBackground({ className = '', size = 400 }: BiteMePieBackgroundProps) {
  return (
    <Image
      src="/images/bite-me-pie.svg"
      alt=""
      width={size}
      height={size}
      className={`opacity-[0.12] mix-blend-darken ${className}`}
      priority
    />
  )
}
