import { Alert, AlertDescription, AlertIcon, AlertTitle, Flex, Text } from '@chakra-ui/react';
import { SeverityLevel } from '../types';

const severityMap: Record<SeverityLevel, { status: 'info' | 'warning' | 'error' | 'success'; title: string }> = {
  normal: { status: 'success', title: 'Nominal' },
  warning: { status: 'warning', title: 'Attention' },
  critical: { status: 'error', title: 'Action Required' },
};

interface AlertBannerProps {
  severity: SeverityLevel;
  messages: string[];
  connectionStatus: string;
}

export const AlertBanner = ({ severity, messages, connectionStatus }: AlertBannerProps) => {
  const config = severityMap[severity] ?? severityMap.normal;
  const message = messages[0] ?? 'System stable and awaiting new tasks.';

  return (
    <Alert
      status={config.status}
      variant="subtle"
      borderRadius="xl"
      borderWidth="1px"
      borderColor="whiteAlpha.200"
      bg="whiteAlpha.100"
      backdropFilter="blur(12px)"
      alignItems="flex-start"
    >
      <AlertIcon boxSize="24px" mr={3} />
      <Flex direction="column" gap={1} flex={1}>
        <AlertTitle textTransform="uppercase" letterSpacing="0.3em" fontSize="xs" color="brand.200">
          {config.title}
        </AlertTitle>
        <AlertDescription>
          <Text fontSize="sm" mb={1} color="slate.100">
            {message}
          </Text>
          <Text fontSize="xs" color="slate.300">
            Link: {connectionStatus}
          </Text>
        </AlertDescription>
      </Flex>
    </Alert>
  );
};
