'use client';

import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { PlusIcon } from 'lucide-react';
import { forwardRef } from 'react';
import type { ControllerRenderProps } from 'react-hook-form';

export const ManagedMCPAgentsField = forwardRef<HTMLDivElement, ControllerRenderProps>(({ value, onChange, name, disabled, onBlur }, ref) => {
  const list = (value as any[] | null) ?? [];
  return (
    <div className="space-y-2" ref={ref}>
      {list.map((item, index) => (
        <div key={index} className="grid grid-cols-6 gap-2 items-center">
          <Input placeholder="name" disabled={disabled} value={item.name ?? ''} onChange={e => {
            const next = [...list];
            next[index] = { ...next[index], name: e.target.value };
            onChange(next);
          }} onBlur={onBlur} />
          <Input placeholder="tidb_host" disabled={disabled} value={item.tidb_host ?? ''} onChange={e => {
            const next = [...list];
            next[index] = { ...next[index], tidb_host: e.target.value };
            onChange(next);
          }} onBlur={onBlur} />
          <Input placeholder="tidb_port" disabled={disabled} value={item.tidb_port ?? ''} onChange={e => {
            const next = [...list];
            next[index] = { ...next[index], tidb_port: e.target.value };
            onChange(next);
          }} onBlur={onBlur} />
          <Input placeholder="tidb_username" disabled={disabled} value={item.tidb_username ?? ''} onChange={e => {
            const next = [...list];
            next[index] = { ...next[index], tidb_username: e.target.value };
            onChange(next);
          }} onBlur={onBlur} />
          <Input type="password" placeholder="tidb_password" disabled={disabled} value={item.tidb_password ?? ''} onChange={e => {
            const next = [...list];
            next[index] = { ...next[index], tidb_password: e.target.value };
            onChange(next);
          }} onBlur={onBlur} />
          <Input placeholder="tidb_database" disabled={disabled} value={item.tidb_database ?? ''} onChange={e => {
            const next = [...list];
            next[index] = { ...next[index], tidb_database: e.target.value };
            onChange(next);
          }} onBlur={onBlur} />

          <div className="col-span-6 flex gap-2">
            <Button variant="secondary" type="button" disabled={disabled} onClick={() => {
              const next = [...list];
              next.splice(index + 1, 0, { name: '', tidb_host: '', tidb_port: '4000', tidb_username: '', tidb_password: '', tidb_database: '' });
              onChange(next);
            }}>
              Add below
            </Button>
            <Button variant="ghost" type="button" disabled={disabled} onClick={() => {
              const next = [...list];
              next.splice(index, 1);
              onChange(next);
            }}>
              Delete
            </Button>
          </div>
        </div>
      ))}

      {!disabled && (
        <Button className="gap-2" variant="ghost" type="button" onClick={() => onChange([...list, { name: '', tidb_host: '', tidb_port: '4000', tidb_username: '', tidb_password: '', tidb_database: '' }])}>
          <PlusIcon className="size-4" />
          Add
        </Button>
      )}
    </div>
  );
});

ManagedMCPAgentsField.displayName = 'ManagedMCPAgentsField';

