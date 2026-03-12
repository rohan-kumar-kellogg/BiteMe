import { chromium } from 'playwright';

const APP_URL = 'http://localhost:3000';
const TEST_USERNAME = 'rohan_demo';
const TEST_EMAIL = 'rohan.demo@example.com';

// Utility to capture console logs
function setupConsoleCapture(page) {
  const logs = [];
  page.on('console', msg => {
    const text = msg.text();
    // Only capture relevant logs
    if (text.includes('[appState]') || 
        text.includes('[session]') || 
        text.includes('[auth]') || 
        text.includes('[upload]') || 
        text.includes('[profile]') ||
        text.includes('[loading render]')) {
      logs.push({ type: msg.type(), text });
    }
  });
  return logs;
}

async function waitForStateTransition(page, timeout = 5000) {
  try {
    await page.waitForLoadState('networkidle', { timeout });
  } catch (e) {
    // Continue if timeout
  }
}

async function sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

// FLOW 1: Fresh load with no session
async function testFlow1(browser) {
  console.log('\n========== FLOW 1: Fresh load with no session ==========');
  const context = await browser.newContext();
  const page = await context.newPage();
  const logs = setupConsoleCapture(page);

  try {
    await page.goto(APP_URL);
    await sleep(2000);

    const loadingVisible = await page.locator('text=Loading your profile').isVisible().catch(() => false);
    const loadingRestoring = await page.locator('text=Restoring your session').isVisible().catch(() => false);
    const welcomeVisible = await page.locator('text=Welcome').first().isVisible().catch(() => false);

    console.log('✓ Page loaded');
    console.log(`  - LoadingState visible: ${loadingVisible || loadingRestoring ? 'YES' : 'NO'}`);
    console.log(`  - WelcomeScreen visible: ${welcomeVisible ? 'YES' : 'NO'}`);
    console.log(`  - Console logs captured: ${logs.length}`);
    
    if (logs.length > 0) {
      console.log('  - Key logs:');
      logs.forEach(log => console.log(`    ${log.text}`));
    }

    return {
      success: welcomeVisible && !loadingVisible && !loadingRestoring,
      stuck: loadingVisible || loadingRestoring,
      logs: logs.map(l => l.text)
    };
  } catch (error) {
    console.log('✗ Error:', error.message);
    return { success: false, stuck: false, error: error.message, logs: logs.map(l => l.text) };
  } finally {
    await context.close();
  }
}

// FLOW 2: Fresh load with saved session
async function testFlow2(browser) {
  console.log('\n========== FLOW 2: Fresh load with saved session ==========');
  const context = await browser.newContext();
  const page = await context.newPage();
  const logs = setupConsoleCapture(page);

  try {
    // First create a session by logging in
    console.log('  Step 1: Creating session by logging in...');
    await page.goto(APP_URL);
    await sleep(1000);

    // Fill login form
    const emailInput = page.locator('input[type="email"], input[placeholder*="email" i]').first();
    const usernameInput = page.locator('input[type="text"], input[placeholder*="username" i]').first();
    
    await emailInput.fill(TEST_EMAIL);
    await usernameInput.fill(TEST_USERNAME);
    
    const submitButton = page.locator('button[type="submit"]').first();
    await submitButton.click();

    console.log('  - Login submitted, waiting for profile...');
    await sleep(5000); // Wait for login to complete

    // Check if we're on profile page
    const profileVisible = await page.locator('text=Profile').first().isVisible().catch(() => false);
    console.log(`  - Profile loaded: ${profileVisible}`);

    // Now hard refresh
    console.log('  Step 2: Hard refreshing page...');
    logs.length = 0; // Clear logs
    await page.reload({ waitUntil: 'networkidle' });
    await sleep(3000);

    const loadingAppeared = logs.some(l => l.text.includes('[loading render]'));
    const restoringVisible = await page.locator('text=Restoring your session').isVisible().catch(() => false);
    const profileStillVisible = await page.locator('text=Profile').first().isVisible().catch(() => false);
    const stuck = await page.locator('text=Loading').isVisible().catch(() => false);

    console.log('✓ Refresh completed');
    console.log(`  - LoadingState appeared: ${loadingAppeared || restoringVisible ? 'YES' : 'NO'}`);
    console.log(`  - Profile restored: ${profileStillVisible ? 'YES' : 'NO'}`);
    console.log(`  - Stuck loading: ${stuck ? 'YES' : 'NO'}`);
    console.log(`  - Console logs: ${logs.length}`);
    
    if (logs.length > 0) {
      console.log('  - Key logs:');
      logs.forEach(log => console.log(`    ${log.text}`));
    }

    return {
      success: profileStillVisible && !stuck,
      stuck: stuck,
      logs: logs.map(l => l.text)
    };
  } catch (error) {
    console.log('✗ Error:', error.message);
    return { success: false, stuck: false, error: error.message, logs: logs.map(l => l.text) };
  } finally {
    await context.close();
  }
}

// FLOW 3: Manual login submit
async function testFlow3(browser) {
  console.log('\n========== FLOW 3: Manual login submit ==========');
  const context = await browser.newContext();
  const page = await context.newPage();
  const logs = setupConsoleCapture(page);

  try {
    await page.goto(APP_URL);
    await sleep(1000);

    console.log('  Filling login form...');
    const emailInput = page.locator('input[type="email"], input[placeholder*="email" i]').first();
    const usernameInput = page.locator('input[type="text"], input[placeholder*="username" i]').first();
    
    await emailInput.fill(TEST_EMAIL);
    await usernameInput.fill(TEST_USERNAME);

    const submitButton = page.locator('button[type="submit"]').first();
    await submitButton.click();
    console.log('  - Form submitted');

    await sleep(1000);
    const loadingVisible = await page.locator('text=Loading your profile').isVisible().catch(() => false);
    console.log(`  - LoadingState visible: ${loadingVisible ? 'YES' : 'NO'}`);

    // Wait for profile or timeout
    await sleep(5000);

    const profileVisible = await page.locator('text=Profile').first().isVisible().catch(() => false);
    const stillLoading = await page.locator('text=Loading').isVisible().catch(() => false);

    console.log('✓ Login flow completed');
    console.log(`  - Profile loaded: ${profileVisible ? 'YES' : 'NO'}`);
    console.log(`  - Stuck loading: ${stillLoading ? 'YES' : 'NO'}`);
    console.log(`  - Console logs: ${logs.length}`);
    
    if (logs.length > 0) {
      console.log('  - Key logs:');
      logs.forEach(log => console.log(`    ${log.text}`));
    }

    return {
      success: profileVisible && !stillLoading,
      stuck: stillLoading,
      logs: logs.map(l => l.text)
    };
  } catch (error) {
    console.log('✗ Error:', error.message);
    return { success: false, stuck: false, error: error.message, logs: logs.map(l => l.text) };
  } finally {
    await context.close();
  }
}

// FLOW 4: Hit back to return to welcome
async function testFlow4(browser) {
  console.log('\n========== FLOW 4: Hit back to return to welcome ==========');
  const context = await browser.newContext();
  const page = await context.newPage();
  const logs = setupConsoleCapture(page);

  try {
    // First login
    console.log('  Step 1: Logging in...');
    await page.goto(APP_URL);
    await sleep(1000);

    const emailInput = page.locator('input[type="email"], input[placeholder*="email" i]').first();
    const usernameInput = page.locator('input[type="text"], input[placeholder*="username" i]').first();
    await emailInput.fill(TEST_EMAIL);
    await usernameInput.fill(TEST_USERNAME);
    
    await page.locator('button[type="submit"]').first().click();
    await sleep(5000);

    // Look for back button
    console.log('  Step 2: Looking for Back button...');
    logs.length = 0; // Clear logs

    // Try various back button selectors
    const backButton = await page.locator('button:has-text("Back"), button:has-text("back"), [aria-label*="back" i]').first();
    const backExists = await backButton.isVisible().catch(() => false);

    if (!backExists) {
      console.log('  ⚠ No Back button found in UI');
      return { success: false, stuck: false, error: 'Back button not found', logs: [] };
    }

    await backButton.click();
    console.log('  - Back button clicked');
    await sleep(2000);

    const welcomeVisible = await page.locator('text=Welcome').first().isVisible().catch(() => false);
    const loadingVisible = await page.locator('text=Loading').isVisible().catch(() => false);

    console.log('✓ Back navigation completed');
    console.log(`  - Welcome screen visible: ${welcomeVisible ? 'YES' : 'NO'}`);
    console.log(`  - Loading state: ${loadingVisible ? 'YES' : 'NO'}`);
    console.log(`  - Console logs: ${logs.length}`);
    
    if (logs.length > 0) {
      console.log('  - Key logs:');
      logs.forEach(log => console.log(`    ${log.text}`));
    }

    return {
      success: welcomeVisible && !loadingVisible,
      stuck: loadingVisible,
      logs: logs.map(l => l.text)
    };
  } catch (error) {
    console.log('✗ Error:', error.message);
    return { success: false, stuck: false, error: error.message, logs: logs.map(l => l.text) };
  } finally {
    await context.close();
  }
}

// FLOW 5: Logout then login again
async function testFlow5(browser) {
  console.log('\n========== FLOW 5: Logout then login again ==========');
  const context = await browser.newContext();
  const page = await context.newPage();
  const logs = setupConsoleCapture(page);

  try {
    // First login
    console.log('  Step 1: Logging in...');
    await page.goto(APP_URL);
    await sleep(1000);

    const emailInput = page.locator('input[type="email"], input[placeholder*="email" i]').first();
    const usernameInput = page.locator('input[type="text"], input[placeholder*="username" i]').first();
    await emailInput.fill(TEST_EMAIL);
    await usernameInput.fill(TEST_USERNAME);
    
    await page.locator('button[type="submit"]').first().click();
    await sleep(5000);

    // Look for logout button
    console.log('  Step 2: Looking for Logout button...');
    const logoutButton = await page.locator('button:has-text("Logout"), button:has-text("logout"), button:has-text("Log out"), [aria-label*="logout" i]').first();
    const logoutExists = await logoutButton.isVisible().catch(() => false);

    if (!logoutExists) {
      console.log('  ⚠ No Logout button found, trying alternative selectors...');
      // Try icon-based logout
      const logoutIcon = await page.locator('[data-testid*="logout"], [aria-label*="logout"]').first();
      const iconExists = await logoutIcon.isVisible().catch(() => false);
      
      if (!iconExists) {
        console.log('  ⚠ No Logout button/icon found in UI');
        return { success: false, stuck: false, error: 'Logout button not found', logs: [] };
      }
      await logoutIcon.click();
    } else {
      await logoutButton.click();
    }

    console.log('  - Logout clicked');
    await sleep(2000);

    const welcomeAfterLogout = await page.locator('text=Welcome').first().isVisible().catch(() => false);
    console.log(`  - Back to welcome: ${welcomeAfterLogout ? 'YES' : 'NO'}`);

    // Re-login
    console.log('  Step 3: Logging in again...');
    logs.length = 0; // Clear logs

    const emailInput2 = page.locator('input[type="email"], input[placeholder*="email" i]').first();
    const usernameInput2 = page.locator('input[type="text"], input[placeholder*="username" i]').first();
    await emailInput2.fill(TEST_EMAIL);
    await usernameInput2.fill(TEST_USERNAME);
    
    await page.locator('button[type="submit"]').first().click();
    await sleep(5000);

    const profileVisible = await page.locator('text=Profile').first().isVisible().catch(() => false);
    const stillLoading = await page.locator('text=Loading').isVisible().catch(() => false);

    console.log('✓ Logout and re-login completed');
    console.log(`  - Re-login successful: ${profileVisible ? 'YES' : 'NO'}`);
    console.log(`  - Stuck loading: ${stillLoading ? 'YES' : 'NO'}`);
    console.log(`  - Console logs: ${logs.length}`);
    
    if (logs.length > 0) {
      console.log('  - Key logs:');
      logs.forEach(log => console.log(`    ${log.text}`));
    }

    return {
      success: profileVisible && !stillLoading,
      stuck: stillLoading,
      logs: logs.map(l => l.text)
    };
  } catch (error) {
    console.log('✗ Error:', error.message);
    return { success: false, stuck: false, error: error.message, logs: logs.map(l => l.text) };
  } finally {
    await context.close();
  }
}

// FLOW 6: Incognito fresh load
async function testFlow6(browser) {
  console.log('\n========== FLOW 6: Incognito fresh load ==========');
  const context = await browser.newContext(); // Fresh context = incognito equivalent
  const page = await context.newPage();
  const logs = setupConsoleCapture(page);

  try {
    await page.goto(APP_URL);
    await sleep(2000);

    const loadingVisible = await page.locator('text=Loading').isVisible().catch(() => false);
    const welcomeVisible = await page.locator('text=Welcome').first().isVisible().catch(() => false);

    console.log('✓ Fresh context loaded');
    console.log(`  - LoadingState visible: ${loadingVisible ? 'YES' : 'NO'}`);
    console.log(`  - WelcomeScreen visible: ${welcomeVisible ? 'YES' : 'NO'}`);
    console.log(`  - Console logs: ${logs.length}`);
    
    if (logs.length > 0) {
      console.log('  - Key logs:');
      logs.forEach(log => console.log(`    ${log.text}`));
    }

    return {
      success: welcomeVisible && !loadingVisible,
      stuck: loadingVisible,
      logs: logs.map(l => l.text)
    };
  } catch (error) {
    console.log('✗ Error:', error.message);
    return { success: false, stuck: false, error: error.message, logs: logs.map(l => l.text) };
  } finally {
    await context.close();
  }
}

// FLOW 7: Upload after login
async function testFlow7(browser) {
  console.log('\n========== FLOW 7: Upload after login ==========');
  const context = await browser.newContext();
  const page = await context.newPage();
  const logs = setupConsoleCapture(page);

  try {
    // First login
    console.log('  Step 1: Logging in...');
    await page.goto(APP_URL);
    await sleep(1000);

    const emailInput = page.locator('input[type="email"], input[placeholder*="email" i]').first();
    const usernameInput = page.locator('input[type="text"], input[placeholder*="username" i]').first();
    await emailInput.fill(TEST_EMAIL);
    await usernameInput.fill(TEST_USERNAME);
    
    await page.locator('button[type="submit"]').first().click();
    await sleep(5000);

    // Look for upload input or button
    console.log('  Step 2: Looking for upload capability...');
    logs.length = 0;

    const fileInput = page.locator('input[type="file"]').first();
    const fileInputExists = await fileInput.isVisible({ timeout: 2000 }).catch(() => false);

    if (!fileInputExists) {
      // Try to find upload button that might reveal file input
      const uploadButton = page.locator('button:has-text("Upload"), button:has-text("upload"), [aria-label*="upload" i]').first();
      const uploadButtonExists = await uploadButton.isVisible().catch(() => false);
      
      if (!uploadButtonExists) {
        console.log('  ⚠ BLOCKER: No file input or upload button found in UI');
        console.log('  - Cannot proceed with upload test without accessible file input');
        return { 
          success: false, 
          stuck: false, 
          error: 'BLOCKER: File upload input not accessible in browser automation environment',
          logs: []
        };
      }
      
      await uploadButton.click();
      await sleep(500);
    }

    // Check if file input is now available
    const fileInputNow = await page.locator('input[type="file"]').first().isVisible().catch(() => false);
    
    if (!fileInputNow) {
      console.log('  ⚠ BLOCKER: File input not accessible after attempting to reveal it');
      return { 
        success: false, 
        stuck: false, 
        error: 'BLOCKER: File upload input not accessible',
        logs: []
      };
    }

    // Create a test image file (1x1 PNG)
    const buffer = Buffer.from(
      'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==',
      'base64'
    );
    
    console.log('  - Attempting file upload...');
    await page.locator('input[type="file"]').first().setInputFiles({
      name: 'test-food.png',
      mimeType: 'image/png',
      buffer: buffer
    });

    await sleep(1000);
    const uploadingVisible = await page.locator('text=Analyzing, text=Uploading').first().isVisible().catch(() => false);
    console.log(`  - Upload initiated: ${uploadingVisible ? 'YES' : 'NO'}`);

    // Wait for upload to complete
    await sleep(8000);

    const stillUploading = await page.locator('text=Analyzing, text=Uploading').first().isVisible().catch(() => false);
    const profileVisible = await page.locator('text=Profile').first().isVisible().catch(() => false);

    console.log('✓ Upload flow completed');
    console.log(`  - Upload completed: ${!stillUploading ? 'YES' : 'NO'}`);
    console.log(`  - Stuck uploading: ${stillUploading ? 'YES' : 'NO'}`);
    console.log(`  - Console logs: ${logs.length}`);
    
    if (logs.length > 0) {
      console.log('  - Key logs:');
      logs.forEach(log => console.log(`    ${log.text}`));
    }

    return {
      success: !stillUploading && profileVisible,
      stuck: stillUploading,
      logs: logs.map(l => l.text)
    };
  } catch (error) {
    console.log('✗ Error:', error.message);
    return { success: false, stuck: false, error: error.message, logs: logs.map(l => l.text) };
  } finally {
    await context.close();
  }
}

// Main execution
(async () => {
  console.log('='.repeat(60));
  console.log('BiteMe App - State/Loading Bug Test Suite');
  console.log('='.repeat(60));
  console.log(`Target URL: ${APP_URL}`);
  console.log(`Test Credentials: ${TEST_EMAIL} / ${TEST_USERNAME}`);
  console.log('='.repeat(60));

  const browser = await chromium.launch({ headless: true });
  
  const results = {
    flow1: await testFlow1(browser),
    flow2: await testFlow2(browser),
    flow3: await testFlow3(browser),
    flow4: await testFlow4(browser),
    flow5: await testFlow5(browser),
    flow6: await testFlow6(browser),
    flow7: await testFlow7(browser),
  };

  await browser.close();

  // Final Report
  console.log('\n\n' + '='.repeat(60));
  console.log('FINAL REPORT');
  console.log('='.repeat(60));

  let anyStuck = false;
  let flowsWithIssues = [];

  Object.entries(results).forEach(([flow, result]) => {
    const flowNum = flow.replace('flow', '');
    const status = result.success ? '✅ PASS' : (result.error ? '⚠️  BLOCKED' : '❌ FAIL');
    console.log(`\nFlow ${flowNum}: ${status}`);
    
    if (result.stuck) {
      anyStuck = true;
      flowsWithIssues.push(flowNum);
      console.log(`  ⚠️  STUCK LOADING DETECTED`);
    }
    
    if (result.error) {
      console.log(`  Error: ${result.error}`);
    }
    
    if (result.logs && result.logs.length > 0) {
      console.log(`  Logs captured: ${result.logs.length}`);
    }
  });

  console.log('\n' + '='.repeat(60));
  console.log('VERDICT');
  console.log('='.repeat(60));
  
  if (anyStuck) {
    console.log(`❌ STUCK LOADING REPRODUCES: YES`);
    console.log(`   Affected flows: ${flowsWithIssues.join(', ')}`);
  } else {
    console.log(`✅ STUCK LOADING REPRODUCES: NO`);
    console.log(`   All flows completed without stuck states`);
  }

  const passCount = Object.values(results).filter(r => r.success).length;
  const totalCount = Object.keys(results).length;
  console.log(`\nTest Summary: ${passCount}/${totalCount} flows passed`);
  console.log('='.repeat(60));
})();
