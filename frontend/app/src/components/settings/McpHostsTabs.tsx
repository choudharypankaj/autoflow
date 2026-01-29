"use client";

import type { AllSettings } from '@/api/site-settings';
import { GrafanaMCPSettings, MCPSettings, ManagedMCPSettings } from '@/components/settings/IntegrationsSettings';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';

export function McpHostsTabs ({ settings }: { settings: AllSettings }) {
  return (
    <Tabs defaultValue="database">
      <TabsList>
        <TabsTrigger value="database">Database</TabsTrigger>
        <TabsTrigger value="grafana">Grafana</TabsTrigger>
      </TabsList>
      <TabsContent value="database">
        <MCPSettings schema={settings} />
        <div className="mt-10" />
        <ManagedMCPSettings schema={settings} />
      </TabsContent>
      <TabsContent value="grafana">
        <GrafanaMCPSettings schema={settings} />
      </TabsContent>
    </Tabs>
  );
}
