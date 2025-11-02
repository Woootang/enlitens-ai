import { expect, test } from '@playwright/test';

test.describe('Mission Control Dashboard smoke tests', () => {
  test('renders summary and pipeline panels', async ({ page }) => {
    await page.goto('/');
    await expect(page.getByText(/Active Document/i)).toBeVisible();
    await expect(page.getByText(/Pipeline progress/i)).toBeVisible();
    await expect(page.getByText(/Agent Performance/i)).toBeVisible();
    await expect(page.getByText(/Quality Signals/i)).toBeVisible();
  });

  test('assistant dock toggles and accepts input', async ({ page }) => {
    await page.goto('/');
    const toggleButton = page.getByRole('button', { name: /assistant/i });
    await toggleButton.click();
    await expect(page.getByText(/Mission Control Assistant/i)).toBeVisible();
    const input = page.getByPlaceholder('Ask a question');
    await input.fill('Status report');
    await page.getByRole('button', { name: /send/i }).click();
    await expect(page.getByText(/Status report/)).toBeVisible();
  });
});

