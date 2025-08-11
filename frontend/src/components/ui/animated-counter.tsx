"use client";

import { motion, useSpring, useTransform } from "framer-motion";
import { useEffect } from "react";

interface AnimatedCounterProps {
  value: number;
  duration?: number;
  className?: string;
  prefix?: string;
  suffix?: string;
}

export function AnimatedCounter({ 
  value, 
  duration = 1, 
  className,
  prefix = "",
  suffix = ""
}: AnimatedCounterProps) {
  const spring = useSpring(0, { duration: duration * 1000 });
  const display = useTransform(spring, (current) => 
    Math.round(current).toLocaleString()
  );

  useEffect(() => {
    spring.set(value);
  }, [spring, value]);

  return (
    <motion.span className={className}>
      {prefix}
      <motion.span>{display}</motion.span>
      {suffix}
    </motion.span>
  );
}

interface AnimatedProgressProps {
  value: number;
  max: number;
  className?: string;
  showPercentage?: boolean;
}

export function AnimatedProgress({ 
  value, 
  max, 
  className,
  showPercentage = false 
}: AnimatedProgressProps) {
  const percentage = max > 0 ? (value / max) * 100 : 0;
  
  return (
    <div className={className}>
      <div className="flex justify-between items-center mb-2">
        <span className="text-sm text-muted-foreground">
          <AnimatedCounter value={value} /> / <AnimatedCounter value={max} />
        </span>
        {showPercentage && (
          <span className="text-sm font-medium">
            <AnimatedCounter value={percentage} suffix="%" />
          </span>
        )}
      </div>
      <div className="w-full bg-secondary rounded-full h-2">
        <motion.div
          className="bg-primary h-2 rounded-full"
          initial={{ width: 0 }}
          animate={{ width: `${percentage}%` }}
          transition={{ duration: 0.5, ease: "easeOut" }}
        />
      </div>
    </div>
  );
}