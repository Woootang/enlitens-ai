import { Alert, AlertDescription, AlertIcon, AlertTitle, HStack, Tag, Text } from '@chakra-ui/react';
import { AlertMessage, ConnectionState, Severity } from '../types';

interface AlertBannerProps {
  severity: Severity;
  messages: AlertMessage[];
  connectionStatus: ConnectionState;
}

const severityToStatus: Record<Severity, 'info' | 'warning' | 'error'> = {
  info: 'info',
  warning: 'warning',
  danger: 'error',
};

export const AlertBanner = ({ severity, messages, connectionStatus }: AlertBannerProps) => (
  <Alert
    status={severityToStatus[severity]}
    variant="subtle"
    borderRadius="xl"
    bg="rgba(15, 23, 42, 0.65)"
    border="1px solid"
    borderColor="whiteAlpha.200"
    backdropFilter="blur(10px)"
  >
    <AlertIcon />
    <HStack spacing={4} align="flex-start" wrap="wrap">
      <AlertTitle fontWeight="semibold" textTransform="uppercase" letterSpacing="0.2em">
        Mission Control
      </AlertTitle>
      <AlertDescription as={HStack} spacing={4} flexWrap="wrap">
        {messages.length ? (
          messages.map((message) => (
            <Text key={message.id} fontSize="sm" color="slate.100">
              {message.text}
            </Text>
          ))
        ) : (
          <Text fontSize="sm" color="slate.200">
            All systems nominal.
          </Text>
        )}
        <Tag size="sm" variant="solid" colorScheme={connectionStatus.status === 'online' ? 'green' : 'orange'}>
          {connectionStatus.status === 'online' ? 'Live Telemetry' : 'Reconnecting'}
        </Tag>
      </AlertDescription>
    </HStack>
  </Alert>
);
