"use client";

import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { PlusIcon } from 'lucide-react';
import { forwardRef } from 'react';
import type { ControllerRenderProps } from 'react-hook-form';

type PrometheusHostItem = {
  name?: string;
  prometheus_url?: string;
  bearer_token?: string;
};

/** Parse full URL into base URL and port for display. */
function parseUrlAndPort(fullUrl: string): { url: string; port: string } {
  const u = (fullUrl || '').trim();
  if (!u) return { url: '', port: '' };
  const match = u.match(/^(https?:\/\/[^:/]+):(\d+)(?:\/|$)/);
  if (match) return { url: match[1], port: match[2] };
  const matchNoScheme = u.match(/^([^:/]+):(\d+)(?:\/|$)/);
  if (matchNoScheme) return { url: matchNoScheme[1], port: matchNoScheme[2] };
  return { url: u, port: '' };
}

/** Build full prometheus_url from base URL and port. */
function buildPrometheusUrl(url: string, port: string): string {
  const u = (url || '').trim();
  const p = (port || '').trim();
  if (!u) return '';
  if (!p) return u;
  const withoutTrailingPort = u.replace(/:(\d+)\/?$/, '');
  return withoutTrailingPort + ':' + p;
}

export const PrometheusHostsField = forwardRef<HTMLDivElement, ControllerRenderProps>(
  ({ value, onChange, name, disabled, onBlur }, ref) => {
    const list = (value as PrometheusHostItem[] | null) ?? [];

    const updateItem = (index: number, updates: Partial<PrometheusHostItem>) => {
      const next = [...list];
      next[index] = { ...next[index], ...updates };
      onChange(next);
    };

    const setUrlAndPort = (index: number, url: string, port: string) => {
      updateItem(index, { prometheus_url: buildPrometheusUrl(url, port) });
    };

    return (
      <div className="space-y-4" ref={ref}>
        {list.map((item, index) => {
          const { url: urlDisplay, port: portDisplay } = parseUrlAndPort(item.prometheus_url ?? '');
          return (
            <div
              key={index}
              className="rounded-lg border border-border bg-muted/30 p-4 space-y-3"
            >
              <div className="grid gap-3 sm:grid-cols-2">
                <div className="space-y-2">
                  <Label htmlFor={`prom-name-${index}`}>Name</Label>
                  <Input
                    id={`prom-name-${index}`}
                    placeholder="e.g. production"
                    disabled={disabled}
                    value={item.name ?? ''}
                    onChange={e => updateItem(index, { name: e.target.value })}
                    onBlur={onBlur}
                  />
                </div>
              </div>
              <div className="grid gap-3 sm:grid-cols-2">
                <div className="space-y-2">
                  <Label htmlFor={`prom-url-${index}`}>URL (host)</Label>
                  <Input
                    id={`prom-url-${index}`}
                    placeholder="http://localhost or https://prometheus.example.com"
                    disabled={disabled}
                    value={urlDisplay}
                    onChange={e => setUrlAndPort(index, e.target.value, portDisplay)}
                    onBlur={onBlur}
                  />
                </div>
                <div className="space-y-2">
                  <Label htmlFor={`prom-port-${index}`}>Port</Label>
                  <Input
                    id={`prom-port-${index}`}
                    type="text"
                    inputMode="numeric"
                    placeholder="9090"
                    disabled={disabled}
                    value={portDisplay}
                    onChange={e => setUrlAndPort(index, urlDisplay, e.target.value)}
                    onBlur={onBlur}
                  />
                </div>
              </div>
              <div className="space-y-2">
                <Label htmlFor={`prom-token-${index}`}>Bearer token (optional)</Label>
                <Input
                  id={`prom-token-${index}`}
                  type="password"
                  placeholder="Optional auth token"
                  disabled={disabled}
                  value={item.bearer_token ?? ''}
                  onChange={e => updateItem(index, { bearer_token: e.target.value })}
                  onBlur={onBlur}
                />
              </div>
              <div className="flex gap-2 pt-1">
                <Button
                  variant="secondary"
                  type="button"
                  size="sm"
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
                  size="sm"
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
          );
        })}

        {!disabled && (
          <Button
            className="gap-2"
            variant="ghost"
            type="button"
            onClick={() => onChange([...list, { name: '', prometheus_url: '', bearer_token: '' }])}
          >
            <PlusIcon className="size-4" />
            Add Prometheus MCP
          </Button>
        )}
      </div>
    );
  },
);

PrometheusHostsField.displayName = 'PrometheusHostsField';
