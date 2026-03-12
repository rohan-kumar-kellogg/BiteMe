#!/usr/bin/env python3
"""
Browser automation script to test 7 flows for state/loading bugs
"""

import time
import json
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import TimeoutException, NoSuchElementException

BASE_URL = "http://localhost:3000"
TEST_EMAIL = "rohan.demo@example.com"
TEST_USERNAME = "rohan_demo"

class FlowTester:
    def __init__(self):
        self.results = []
        self.driver = None
        
    def setup_driver(self, incognito=False):
        """Setup Chrome driver"""
        options = Options()
        if incognito:
            options.add_argument("--incognito")
        options.add_argument("--enable-logging")
        options.add_argument("--v=1")
        # Enable browser console log capture
        options.set_capability('goog:loggingPrefs', {'browser': 'ALL'})
        
        self.driver = webdriver.Chrome(options=options)
        self.driver.implicitly_wait(2)
        
    def get_console_logs(self):
        """Capture browser console logs"""
        try:
            logs = self.driver.get_log('browser')
            relevant_logs = []
            for log in logs:
                msg = log.get('message', '')
                if any(tag in msg for tag in ['[appState]', '[session]', '[loading render]', 
                                               '[auth]', '[upload]', '[profile]']):
                    relevant_logs.append(msg)
            return relevant_logs
        except Exception as e:
            return [f"Error getting logs: {str(e)}"]
    
    def get_local_storage(self):
        """Get localStorage content"""
        try:
            return self.driver.execute_script("return JSON.stringify(localStorage);")
        except:
            return "{}"
    
    def clear_local_storage(self):
        """Clear localStorage"""
        try:
            self.driver.execute_script("localStorage.clear();")
            return True
        except:
            return False
    
    def wait_for_element(self, by, value, timeout=10):
        """Wait for element to be present"""
        try:
            element = WebDriverWait(self.driver, timeout).until(
                EC.presence_of_element_located((by, value))
            )
            return element
        except TimeoutException:
            return None
    
    def check_for_loading_state(self):
        """Check if LoadingState component is visible"""
        try:
            # Look for loading indicators
            loading_texts = ["Loading your profile", "Restoring your session"]
            for text in loading_texts:
                elements = self.driver.find_elements(By.XPATH, f"//*[contains(text(), '{text}')]")
                if elements:
                    return True, text
            return False, None
        except:
            return False, None
    
    def check_for_stuck_loading(self, max_wait=15):
        """Check if loading state is stuck"""
        start = time.time()
        stuck = False
        
        while time.time() - start < max_wait:
            is_loading, text = self.check_for_loading_state()
            if not is_loading:
                return False
            time.sleep(1)
        
        # Still loading after max_wait seconds = stuck
        return True
    
    def record_result(self, flow_name, success, details):
        """Record test result"""
        self.results.append({
            'flow': flow_name,
            'success': success,
            'timestamp': time.time(),
            'details': details
        })
        print(f"\n{'='*60}")
        print(f"FLOW: {flow_name}")
        print(f"SUCCESS: {success}")
        print(f"DETAILS: {json.dumps(details, indent=2)}")
        print(f"{'='*60}\n")
    
    # FLOW 1: Fresh load with no session
    def flow1_fresh_no_session(self):
        print("\n🧪 FLOW 1: Fresh load with no session")
        try:
            self.setup_driver()
            self.clear_local_storage()
            
            start_time = time.time()
            self.driver.get(BASE_URL)
            time.sleep(2)
            
            logs = self.get_console_logs()
            is_loading, _ = self.check_for_loading_state()
            stuck = self.check_for_stuck_loading(max_wait=5)
            
            # Check if welcome screen is visible
            welcome_visible = self.wait_for_element(By.XPATH, "//*[contains(text(), 'BiteMe') or contains(text(), 'username') or contains(text(), 'email')]", timeout=5) is not None
            
            elapsed = time.time() - start_time
            
            self.record_result('Flow 1: Fresh load no session', not stuck and welcome_visible, {
                'elapsed_seconds': round(elapsed, 2),
                'loading_state_appeared': is_loading,
                'stuck': stuck,
                'welcome_screen_visible': welcome_visible,
                'console_logs': logs[:10] if logs else [],
                'storage': self.get_local_storage()
            })
            
        except Exception as e:
            self.record_result('Flow 1: Fresh load no session', False, {'error': str(e)})
        finally:
            if self.driver:
                self.driver.quit()
    
    # FLOW 2: Fresh load with saved session
    def flow2_fresh_with_session(self):
        print("\n🧪 FLOW 2: Fresh load with saved session")
        try:
            self.setup_driver()
            self.clear_local_storage()
            
            # First login to create session
            self.driver.get(BASE_URL)
            time.sleep(2)
            
            # Fill in credentials
            try:
                email_input = self.wait_for_element(By.XPATH, "//input[@type='email' or @placeholder='Email' or contains(@placeholder, 'email')]")
                username_input = self.wait_for_element(By.XPATH, "//input[@placeholder='Username' or contains(@placeholder, 'username')]")
                
                if email_input:
                    email_input.send_keys(TEST_EMAIL)
                if username_input:
                    username_input.send_keys(TEST_USERNAME)
                
                # Submit
                submit_btn = self.wait_for_element(By.XPATH, "//button[@type='submit' or contains(text(), 'Start') or contains(text(), 'Submit')]")
                if submit_btn:
                    submit_btn.click()
                
                # Wait for profile to load
                time.sleep(5)
                
                # Now hard refresh
                start_time = time.time()
                self.driver.refresh()
                
                time.sleep(2)
                logs = self.get_console_logs()
                is_loading, loading_text = self.check_for_loading_state()
                stuck = self.check_for_stuck_loading(max_wait=10)
                
                elapsed = time.time() - start_time
                
                self.record_result('Flow 2: Fresh load with session', not stuck, {
                    'elapsed_seconds': round(elapsed, 2),
                    'loading_state_appeared': is_loading,
                    'loading_text': loading_text,
                    'stuck': stuck,
                    'console_logs': logs[:15] if logs else [],
                    'storage': self.get_local_storage()
                })
                
            except Exception as e:
                self.record_result('Flow 2: Fresh load with session', False, {'error': str(e), 'stage': 'login/refresh'})
                
        except Exception as e:
            self.record_result('Flow 2: Fresh load with session', False, {'error': str(e), 'stage': 'setup'})
        finally:
            if self.driver:
                self.driver.quit()
    
    # FLOW 3: Manual login submit
    def flow3_manual_login(self):
        print("\n🧪 FLOW 3: Manual login submit")
        try:
            self.setup_driver()
            self.clear_local_storage()
            self.driver.get(BASE_URL)
            time.sleep(2)
            
            start_time = time.time()
            
            # Fill credentials
            email_input = self.wait_for_element(By.XPATH, "//input[@type='email' or @placeholder='Email']")
            username_input = self.wait_for_element(By.XPATH, "//input[@placeholder='Username']")
            
            if email_input:
                email_input.send_keys(TEST_EMAIL)
            if username_input:
                username_input.send_keys(TEST_USERNAME)
            
            # Submit
            submit_btn = self.wait_for_element(By.XPATH, "//button[@type='submit']")
            if submit_btn:
                submit_btn.click()
            
            time.sleep(2)
            logs = self.get_console_logs()
            is_loading, loading_text = self.check_for_loading_state()
            stuck = self.check_for_stuck_loading(max_wait=10)
            
            elapsed = time.time() - start_time
            
            self.record_result('Flow 3: Manual login', not stuck, {
                'elapsed_seconds': round(elapsed, 2),
                'loading_state_appeared': is_loading,
                'loading_text': loading_text,
                'stuck': stuck,
                'console_logs': logs[:15] if logs else []
            })
            
        except Exception as e:
            self.record_result('Flow 3: Manual login', False, {'error': str(e)})
        finally:
            if self.driver:
                self.driver.quit()
    
    # FLOW 4: Hit back to return to welcome
    def flow4_back_button(self):
        print("\n🧪 FLOW 4: Hit back to return to welcome")
        try:
            self.setup_driver()
            # Login first
            self.clear_local_storage()
            self.driver.get(BASE_URL)
            time.sleep(2)
            
            # Quick login
            email_input = self.wait_for_element(By.XPATH, "//input[@type='email']")
            username_input = self.wait_for_element(By.XPATH, "//input[@placeholder='Username']")
            if email_input:
                email_input.send_keys(TEST_EMAIL)
            if username_input:
                username_input.send_keys(TEST_USERNAME)
            
            submit_btn = self.wait_for_element(By.XPATH, "//button[@type='submit']")
            if submit_btn:
                submit_btn.click()
            
            time.sleep(5)  # Wait for profile load
            
            # Now click back
            start_time = time.time()
            back_btn = self.wait_for_element(By.XPATH, "//button[contains(text(), 'Back') or @aria-label='Back']", timeout=5)
            
            if back_btn:
                back_btn.click()
                time.sleep(2)
                
                logs = self.get_console_logs()
                is_loading, _ = self.check_for_loading_state()
                welcome_visible = self.wait_for_element(By.XPATH, "//*[contains(text(), 'BiteMe')]", timeout=5) is not None
                
                elapsed = time.time() - start_time
                
                self.record_result('Flow 4: Back button', welcome_visible, {
                    'elapsed_seconds': round(elapsed, 2),
                    'loading_state_appeared': is_loading,
                    'welcome_visible': welcome_visible,
                    'console_logs': logs[-10:] if logs else []
                })
            else:
                self.record_result('Flow 4: Back button', False, {'error': 'Back button not found'})
                
        except Exception as e:
            self.record_result('Flow 4: Back button', False, {'error': str(e)})
        finally:
            if self.driver:
                self.driver.quit()
    
    # FLOW 5: Logout then login again
    def flow5_logout_relogin(self):
        print("\n🧪 FLOW 5: Logout then login again")
        try:
            self.setup_driver()
            # Login first
            self.clear_local_storage()
            self.driver.get(BASE_URL)
            time.sleep(2)
            
            # Login
            email_input = self.wait_for_element(By.XPATH, "//input[@type='email']")
            username_input = self.wait_for_element(By.XPATH, "//input[@placeholder='Username']")
            if email_input:
                email_input.send_keys(TEST_EMAIL)
            if username_input:
                username_input.send_keys(TEST_USERNAME)
            
            submit_btn = self.wait_for_element(By.XPATH, "//button[@type='submit']")
            if submit_btn:
                submit_btn.click()
            
            time.sleep(5)
            
            # Logout
            logout_btn = self.wait_for_element(By.XPATH, "//button[contains(text(), 'Logout') or @aria-label='Logout' or contains(@class, 'logout')]", timeout=5)
            
            if logout_btn:
                logout_btn.click()
                time.sleep(2)
                
                # Re-login
                start_time = time.time()
                email_input = self.wait_for_element(By.XPATH, "//input[@type='email']")
                username_input = self.wait_for_element(By.XPATH, "//input[@placeholder='Username']")
                if email_input:
                    email_input.send_keys(TEST_EMAIL)
                if username_input:
                    username_input.send_keys(TEST_USERNAME)
                
                submit_btn = self.wait_for_element(By.XPATH, "//button[@type='submit']")
                if submit_btn:
                    submit_btn.click()
                
                time.sleep(2)
                logs = self.get_console_logs()
                is_loading, _ = self.check_for_loading_state()
                stuck = self.check_for_stuck_loading(max_wait=10)
                
                elapsed = time.time() - start_time
                
                self.record_result('Flow 5: Logout + Re-login', not stuck, {
                    'elapsed_seconds': round(elapsed, 2),
                    'loading_state_appeared': is_loading,
                    'stuck': stuck,
                    'console_logs': logs[-15:] if logs else []
                })
            else:
                self.record_result('Flow 5: Logout + Re-login', False, {'error': 'Logout button not found'})
                
        except Exception as e:
            self.record_result('Flow 5: Logout + Re-login', False, {'error': str(e)})
        finally:
            if self.driver:
                self.driver.quit()
    
    # FLOW 6: Incognito fresh load
    def flow6_incognito(self):
        print("\n🧪 FLOW 6: Incognito fresh load")
        try:
            self.setup_driver(incognito=True)
            
            start_time = time.time()
            self.driver.get(BASE_URL)
            time.sleep(2)
            
            logs = self.get_console_logs()
            is_loading, _ = self.check_for_loading_state()
            stuck = self.check_for_stuck_loading(max_wait=5)
            welcome_visible = self.wait_for_element(By.XPATH, "//*[contains(text(), 'BiteMe')]", timeout=5) is not None
            
            elapsed = time.time() - start_time
            
            self.record_result('Flow 6: Incognito load', not stuck and welcome_visible, {
                'elapsed_seconds': round(elapsed, 2),
                'loading_state_appeared': is_loading,
                'stuck': stuck,
                'welcome_visible': welcome_visible,
                'console_logs': logs[:10] if logs else [],
                'storage': self.get_local_storage()
            })
            
        except Exception as e:
            self.record_result('Flow 6: Incognito load', False, {'error': str(e)})
        finally:
            if self.driver:
                self.driver.quit()
    
    # FLOW 7: Upload after login
    def flow7_upload(self):
        print("\n🧪 FLOW 7: Upload after login")
        try:
            self.setup_driver()
            # Login first
            self.clear_local_storage()
            self.driver.get(BASE_URL)
            time.sleep(2)
            
            # Login
            email_input = self.wait_for_element(By.XPATH, "//input[@type='email']")
            username_input = self.wait_for_element(By.XPATH, "//input[@placeholder='Username']")
            if email_input:
                email_input.send_keys(TEST_EMAIL)
            if username_input:
                username_input.send_keys(TEST_USERNAME)
            
            submit_btn = self.wait_for_element(By.XPATH, "//button[@type='submit']")
            if submit_btn:
                submit_btn.click()
            
            time.sleep(5)
            
            # Try to find upload input
            upload_input = self.wait_for_element(By.XPATH, "//input[@type='file']", timeout=5)
            
            if upload_input:
                # Check if we have a test image
                import os
                test_image_paths = [
                    '/Users/rohankumar/PycharmProjects/BiteMe/data/images/test.jpg',
                    '/Users/rohankumar/PycharmProjects/BiteMe/data/images/American/test.jpg',
                ]
                
                test_image = None
                for path in test_image_paths:
                    if os.path.exists(path):
                        test_image = path
                        break
                
                if test_image:
                    start_time = time.time()
                    upload_input.send_keys(test_image)
                    time.sleep(3)
                    
                    logs = self.get_console_logs()
                    stuck = self.check_for_stuck_loading(max_wait=15)
                    
                    elapsed = time.time() - start_time
                    
                    self.record_result('Flow 7: Upload', not stuck, {
                        'elapsed_seconds': round(elapsed, 2),
                        'stuck': stuck,
                        'file_used': test_image,
                        'console_logs': logs[-15:] if logs else []
                    })
                else:
                    self.record_result('Flow 7: Upload', False, {
                        'blocker': 'No test image file found',
                        'searched_paths': test_image_paths
                    })
            else:
                self.record_result('Flow 7: Upload', False, {
                    'blocker': 'Upload input element not found in DOM',
                    'page_source_sample': self.driver.page_source[:500]
                })
                
        except Exception as e:
            self.record_result('Flow 7: Upload', False, {'error': str(e)})
        finally:
            if self.driver:
                self.driver.quit()
    
    def run_all_flows(self):
        """Run all 7 flows"""
        print("\n" + "="*60)
        print("🚀 STARTING BROWSER FLOW TESTS")
        print("="*60)
        
        self.flow1_fresh_no_session()
        self.flow2_fresh_with_session()
        self.flow3_manual_login()
        self.flow4_back_button()
        self.flow5_logout_relogin()
        self.flow6_incognito()
        self.flow7_upload()
        
        # Final report
        print("\n" + "="*60)
        print("📊 FINAL REPORT")
        print("="*60)
        
        stuck_loading_found = False
        for result in self.results:
            status = "✅ PASS" if result['success'] else "❌ FAIL"
            print(f"{status} - {result['flow']}")
            
            if result['details'].get('stuck'):
                stuck_loading_found = True
                print(f"   ⚠️  STUCK LOADING DETECTED!")
        
        print("\n" + "="*60)
        print("🔍 FINAL VERDICT")
        print("="*60)
        if stuck_loading_found:
            print("❌ STUCK LOADING BUG REPRODUCED")
            print("   Flows with stuck loading:")
            for result in self.results:
                if result['details'].get('stuck'):
                    print(f"   - {result['flow']}")
        else:
            print("✅ NO STUCK LOADING DETECTED")
            print("   All flows transitioned correctly")
        
        # Save detailed results
        with open('/Users/rohankumar/PycharmProjects/BiteMe/test_results.json', 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"\n📝 Detailed results saved to: test_results.json")


if __name__ == "__main__":
    tester = FlowTester()
    tester.run_all_flows()
