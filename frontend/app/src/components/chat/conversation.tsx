'use client';

import type { Chat, ChatMessage } from '@/api/chats';
import type { ChatController } from '@/components/chat/chat-controller';
import { ChatControllerProvider, useChatController, useChatMessageControllers, useChatMessageGroups, useChatPostState } from '@/components/chat/chat-hooks';
import { ConversationMessageGroups } from '@/components/chat/conversation-message-groups';
import { MessageInput } from '@/components/chat/message-input';
import { SecuritySettingContext, withReCaptcha } from '@/components/security-setting-provider';
import { useSize } from '@/components/use-size';
import { cn } from '@/lib/utils';
import { type ChangeEvent, type FormEvent, type ReactNode, type Ref, useContext, useEffect, useImperativeHandle, useState } from 'react';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { getMcpAgents } from '@/api/mcp';

export interface ConversationProps {
  chatId?: string;

  className?: string;
  open: boolean;
  chat: Chat | undefined;
  history: ChatMessage[];

  /* Only for widgets */
  placeholder?: (controller: ChatController, postState: ReturnType<typeof useChatPostState>) => ReactNode;
  preventMutateBrowserHistory?: boolean;
  preventShiftMessageInput?: boolean;
  newChatRef?: Ref<ChatController['post'] | undefined>;
}

export function Conversation ({ open, chat, chatId, history, placeholder, preventMutateBrowserHistory = false, preventShiftMessageInput = false, newChatRef, className }: ConversationProps) {
  const [inputElement, setInputElement] = useState<HTMLTextAreaElement | null>(null);

  const controller = useChatController(chatId, chat, history, inputElement);
  const postState = useChatPostState(controller);
  const groups = useChatMessageGroups(useChatMessageControllers(controller));

  const [input, setInput] = useState('');
  const handleInputChange = (e: ChangeEvent<HTMLTextAreaElement>) => {
    setInput(e.target.value);
  };

  const { ref, size } = useSize();

  const security = useContext(SecuritySettingContext);
  const [agentNames, setAgentNames] = useState<string[]>([]);
  const [selectedAgent, setSelectedAgent] = useState<string | undefined>(undefined);
  useEffect(() => {
    getMcpAgents().then(names => {
      setAgentNames(names);
      if (!selectedAgent && names.length) {
        setSelectedAgent(names[0]);
      }
    }).catch(() => {});
  }, []);

  const submitWithReCaptcha = async (e: FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    withReCaptcha({
      action: 'chat',
      siteKey: security?.google_recaptcha_site_key || '',
      mode: security?.google_recaptcha,
    }, ({ token, action }) => {
      const headers: Record<string, string> = {
        'X-Recaptcha-Token': token,
        'X-Recaptcha-Action': action,
      };
      if (selectedAgent) {
        headers['X-MCP-Host-Name'] = selectedAgent;
      }
      controller.post({
        content: input,
        headers,
      });
      setInput('');
    });
  };

  const disabled = !!postState.params;
  const actionDisabled = disabled || !input.trim();

  useImperativeHandle(newChatRef, () => {
    return controller.post.bind(controller);
  }, [controller]);

  return (
    <ChatControllerProvider controller={controller}>
      {!postState.params && !groups.length && placeholder?.(controller, postState)}
      <div ref={ref} className={cn(
        'mx-auto space-y-4 transition-all relative md:max-w-screen-md md:min-h-screen md:p-body',
        className,
      )}>
        {!!agentNames?.length && (
          <div className="sticky top-[4.5rem] z-10 flex justify-end">
            <div className="bg-background/70 backdrop-blur border rounded px-2 py-1">
              <Select value={selectedAgent} onValueChange={setSelectedAgent}>
                <SelectTrigger className="w-56 h-8">
                  <SelectValue placeholder="Select DB Agent" />
                </SelectTrigger>
                <SelectContent>
                  {agentNames.map((name: string) => (
                    <SelectItem key={name} value={name}>{name}</SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
          </div>
        )}
        <ConversationMessageGroups groups={groups} />
        <div className="h-24"></div>
      </div>
      {size && open && <form className={cn('block h-max p-4 fixed bottom-0', preventShiftMessageInput && 'absolute pb-0')} onSubmit={submitWithReCaptcha} style={{ left: (preventShiftMessageInput ? 0 : size.x) + 16, width: size.width - 32 }}>
        <MessageInput inputRef={setInputElement} className="w-full transition-all" disabled={disabled} actionDisabled={actionDisabled} inputProps={{ value: input, onChange: handleInputChange, disabled }} />
      </form>}
    </ChatControllerProvider>
  );
}
