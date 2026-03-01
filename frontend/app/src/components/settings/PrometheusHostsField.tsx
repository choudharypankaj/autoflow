"use client";

import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { PlusIcon } from 'lucide-react';
import { forwardRef } from 'react';
import type { ControllerRenderProps } from 'react-hook-form';

export const PrometheusHostsField = forwardRef<HTMLDivElement, ControllerRenderProps>(
  ({ value, onChange, name, disabled, onBlur }, ref) => {
    const list = (value as { name?: string; prometheus_url?: string; bearer_token?: string }[] | null) ?? [];
    return (
      <div className="space-y-2" ref={ref}>
        {list.map((item, index) => (
          <div key={index} className="grid grid-cols-3 gap-2 items-center">
            <Input
              placeholder="name"
              disabled={disabled}
              value={item.name ?? ''}
              onChange={e => {
                const next = [...list];
                next[index] = { ...next[index], name: e.target.value };
                onChange(next);
              }}
              onBlur={onBlur}
            />
            <Input
              placeholder="prometheus_url"
              disabled={disabled}
              value={item.prometheus_url ?? ''}
              onChange={e => {
                const next = [...list];
                next[index] = { ...next[index], prometheus_url: e.target.value };
                onChange(next);
              }}
              onBlur={onBlur}
            />
            <Input
              type="password"
              placeholder="bearer_token (optional)"
              disabled={disabled}
              value={item.bearer_token ?? ''}
              onChange={e => {
                const next = [...list];
                next[index] = { ...next[index], bearer_token: e.target.value };
                onChange(next);
              }}
              onBlur={onBlur}
            />

            <div className="col-span-3 flex gap-2">
              <Button
                variant="secondary"
                type="button"
                disabled={disabled}
                onClick={() => {
                  const next = [...list];
                  next.splice(index + 1, 0, { name: '', prometheus_url: '', bearer_token: '' });
                  onChange(next);
                }}
              >
                Add below
              </Button>
              <Button
                variant="ghost"
                type="button"
                disabled={disabled}
                onClick={() => {
                  const next = [...list];
                  next.splice(index, 1);
                  onChange(next);
                }}
              >
                Delete
              </Button>
            </div>
          </div>
        ))}

        {!disabled && (
          <Button
            className="gap-2"
            variant="ghost"
            type="button"
            onClick={() => onChange([...list, { name: '', prometheus_url: '', bearer_token: '' }])}
          >
            <PlusIcon className="size-4" />
            Add
          </Button>
        )}
      </div>
    );
  },
);

PrometheusHostsField.displayName = 'PrometheusHostsField';
