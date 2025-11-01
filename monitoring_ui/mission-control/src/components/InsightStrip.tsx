import { Card, CardBody, Heading, Tag, Text, Wrap, WrapItem } from '@chakra-ui/react';
import { DashboardInsight } from '../types';

interface InsightStripProps {
  insights: DashboardInsight[];
}

const severityColor: Record<DashboardInsight['severity'], string> = {
  info: 'blue',
  normal: 'green',
  warning: 'yellow',
  critical: 'red',
};

export const InsightStrip = ({ insights }: InsightStripProps) => {
  return (
    <Card borderRadius="xl" borderColor="whiteAlpha.200">
      <CardBody display="flex" flexDirection="column" gap={4}>
        <Heading as="h3" size="xs" textTransform="uppercase" letterSpacing="0.3em" color="brand.200">
          Insights
        </Heading>
        {insights.length === 0 ? (
          <Text fontSize="sm" color="slate.300">
            System nominal. No anomalies detected.
          </Text>
        ) : (
          <Wrap spacing={2}>
            {insights.map((insight) => (
              <WrapItem key={insight.id}>
                <Tag
                  size="lg"
                  borderRadius="full"
                  colorScheme={severityColor[insight.severity] ?? 'blue'}
                  px={3}
                  py={1}
                >
                  {insight.message}
                </Tag>
              </WrapItem>
            ))}
          </Wrap>
        )}
      </CardBody>
    </Card>
  );
};
