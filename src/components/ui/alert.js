// src/components/ui/alert.js
import React from 'react';

export function Alert({ children, variant = 'default', ...props }) {
  const variantStyles = {
    default: 'bg-white border border-gray-200 text-gray-900',
    destructive: 'bg-red-50 border border-red-200 text-red-900'
  };

  return (
    <div 
      className={`p-4 rounded-lg ${variantStyles[variant]} ${props.className}`}
      {...props}
    >
      {children}
    </div>
  );
}

export function AlertTitle({ children, ...props }) {
  return (
    <h5 className="font-semibold mb-2" {...props}>
      {children}
    </h5>
  );
}

export function AlertDescription({ children, ...props }) {
  return (
    <p className="text-sm" {...props}>
      {children}
    </p>
  );
}