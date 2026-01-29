"use client";

import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { PlusIcon } from 'lucide-react';
import { forwardRef } from 'react';
import type { ControllerRenderProps } from 'react-hook-form';

export const GrafanaMCPHostsField = forwardRef<HTMLDivElement, ControllerRenderProps>(
  ({ value, onChange, name, disabled, onBlur }, ref) => {
    const list = (value as any[] | null) ?? [];
    return (
      <div className="space-y-2" ref={ref}>
        {list.map((item, index) => (
          <div key={index} className="grid grid-cols-4 gap-2 items-center">
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
              placeholder="grafana_url"
              disabled={disabled}
              value={item.grafana_url ?? ''}
              onChange={e => {
                const next = [...list];
                next[index] = { ...next[index], grafana_url: e.target.value };
                onChange(next);
              }}
              onBlur={onBlur}
            />
            <Input
              type="password"
              placeholder="grafana_api_key"
              disabled={disabled}
              value={item.grafana_api_key ?? ''}
              onChange={e => {
                const next = [...list];
                next[index] = { ...next[index], grafana_api_key: e.target.value };
                onChange(next);
              }}
              onBlur={onBlur}
            />
            <Input
              placeholder="mcp_ws_url (optional)"
              disabled={disabled}
              value={item.mcp_ws_url ?? ''}
              onChange={e => {
                const next = [...list];
                next[index] = { ...next[index], mcp_ws_url: e.target.value };
                onChange(next);
              }}
              onBlur={onBlur}
            />

            <div className="col-span-4 flex gap-2">
              <Button
                variant="secondary"
                type="button"
                disabled={disabled}
                onClick={() => {
                  const next = [...list];
                  next.splice(index + 1, 0, { name: '', grafana_url: '', grafana_api_key: '', mcp_ws_url: '' });
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
            onClick={() => onChange([...list, { name: '', grafana_url: '', grafana_api_key: '', mcp_ws_url: '' }])}
          >
            <PlusIcon className="size-4" />
            Add
          </Button>
        )}
      </div>
    );
  },
);

GrafanaMCPHostsField.displayName = 'GrafanaMCPHostsField';
