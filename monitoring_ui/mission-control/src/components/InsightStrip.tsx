import { Badge, Card, CardBody, HStack, Icon, Stack, Text } from '@chakra-ui/react';
import { FiAlertTriangle, FiInfo } from 'react-icons/fi';
import { InsightMessage } from '../types';

interface InsightStripProps {
  insights: InsightMessage[];
}

const severityIcon = {
  info: FiInfo,
  warning: FiAlertTriangle,
  danger: FiAlertTriangle,
} as const;

const severityColor = {
  info: 'blue.300',
  warning: 'yellow.300',
  danger: 'red.300',
} as const;

export const InsightStrip = ({ insights }: InsightStripProps) => (
  <Card borderRadius="xl" borderColor="whiteAlpha.200" bg="rgba(15, 23, 42, 0.7)" backdropFilter="blur(10px)">
    <CardBody>
      <Stack spacing={3}>
        <Text textTransform="uppercase" letterSpacing="0.3em" fontSize="xs" color="brand.200">
          Insights
        </Text>
        {insights.length === 0 ? (
          <Text fontSize="sm" color="slate.200">
            No anomalies detected in the last 15 minutes.
          </Text>
        ) : (
          insights.slice(0, 3).map((insight) => (
            <HStack key={insight.id} spacing={3} align="center">
              <Icon as={severityIcon[insight.severity]} color={severityColor[insight.severity]} boxSize={4} />
              <Text fontSize="sm" color="slate.100">
                {insight.message}
              </Text>
              <Badge variant="outline" borderColor="whiteAlpha.300" color="slate.300">
                {new Date(insight.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
              </Badge>
            </HStack>
          ))
        )}
      </Stack>
    </CardBody>
  </Card>
);
