'use client';

import type { AllSettings } from '@/api/site-settings';
import { SettingsField } from '@/components/settings/SettingsField';
import { LinkArrayField } from '@/components/settings/LinkArrayField';
import { z } from 'zod';
import { ManagedMCPAgentsField } from '@/components/settings/ManagedMCPAgentsField';

export function IntegrationsSettings ({ schema, showPostVerificationSettings }: { schema: AllSettings, showPostVerificationSettings: boolean }) {
  return (
    <div className="space-y-8 max-w-screen-md">
      <LangfuseSettings schema={schema} />
      <ManagedMCPSettings schema={schema} />
      {showPostVerificationSettings && <ExperimentalPostVerificationSettings schema={schema} />}
    </div>
  );
}

export function LangfuseSettings ({ schema, hideTitle, disabled, onChanged }: { schema: AllSettings, hideTitle?: boolean, disabled?: boolean, onChanged?: () => void }) {
  return (
    <section className="space-y-6">
      {!hideTitle && <h2 className="text-lg font-medium">Langfuse</h2>}
      <SettingsField name="langfuse_public_key" item={schema.langfuse_public_key} onChanged={onChanged} disabled={disabled} />
      <SettingsField name="langfuse_secret_key" item={schema.langfuse_secret_key} onChanged={onChanged} disabled={disabled} />
      <SettingsField name="langfuse_host" item={schema.langfuse_host} onChanged={onChanged} disabled={disabled} />
    </section>
  );
}

export function MCPSettings ({ schema, hideTitle, disabled, onChanged }: { schema: AllSettings, hideTitle?: boolean, disabled?: boolean, onChanged?: () => void }) {
  return (
    <section className="space-y-6">
      {!hideTitle && <h2 className="text-lg font-medium">MCP</h2>}
      <SettingsField name="mcp_host" item={schema.mcp_host} onChanged={onChanged} disabled={disabled} />
      <SettingsField
        name="mcp_hosts"
        item={schema.mcp_hosts}
        arrayItemSchema={z.object({ text: z.string(), href: z.string() })}
        onChanged={onChanged}
        disabled={disabled}
      >
        {props => <LinkArrayField {...props} />}
      </SettingsField>
    </section>
  );
}

export function ManagedMCPSettings ({ schema, hideTitle, disabled, onChanged }: { schema: AllSettings, hideTitle?: boolean, disabled?: boolean, onChanged?: () => void }) {
  return (
    <section className="space-y-6">
      {!hideTitle && <h2 className="text-lg font-medium">Managed MCP Agents (server-spawned)</h2>}
      <SettingsField
        name="managed_mcp_agents"
        item={schema.managed_mcp_agents}
        arrayItemSchema={z.object({
          name: z.string(),
          tidb_host: z.string(),
          tidb_port: z.string(),
          tidb_username: z.string(),
          tidb_password: z.string(),
          tidb_database: z.string(),
        })}
        onChanged={onChanged}
        disabled={disabled}
      >
        {props => <ManagedMCPAgentsField {...props} />}
      </SettingsField>
      <p className="text-xs text-muted-foreground">Note: Only the agent names are used in the chat dropdown; credentials stay server-side.</p>
    </section>
  );
}

export function ExperimentalPostVerificationSettings ({ schema, hideTitle, disabled, onChanged }: { schema: AllSettings, hideTitle?: boolean, disabled?: boolean, onChanged?: () => void }) {
  return (
    <section className="space-y-6">
      {!hideTitle && <h2 className="text-lg font-medium">[Experimental] Post verifications</h2>}
      <SettingsField name="enable_post_verifications" item={schema.enable_post_verifications} onChanged={onChanged} disabled={disabled} />
      <SettingsField name="enable_post_verifications_for_widgets" item={schema.enable_post_verifications_for_widgets} onChanged={onChanged} disabled={disabled} />
    </section>
  );
}
