#!/usr/bin/env python3
"""
Train Ticket Scraper

This script scrapes train ticket information from:
- Greater Anglia (greateranglia.co.uk)
- National Rail (nationalrail.co.uk)
- The Trainline (thetrainline.com)

The script takes the following parameters:
- Origin station
- Destination station
- Date of journey
- Time of journey 
- Railcard (True/False)

It then returns links to the cheapest tickets along with prices.
"""

import argparse
import time
import datetime
import re
import json
import logging
from urllib.parse import quote, urlencode

import requests
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException, ElementNotInteractableException, StaleElementReferenceException
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TrainTicketScraper:
    """Base class for train ticket scrapers."""
    
    def __init__(self, origin, destination, date, time_of_journey, railcard=False):
        """Initialize with search parameters."""
        self.origin = origin
        self.destination = destination
        self.date = date
        self.time_of_journey = time_of_journey
        self.railcard = railcard
        self.cheapest_ticket = {"price": float('inf'), "link": None}
        
        # Format date to be consistent (YYYY-MM-DD)
        if isinstance(self.date, str):
            try:
                # Try to parse various date formats
                for fmt in ('%Y-%m-%d', '%d/%m/%Y', '%d-%m-%Y', '%m/%d/%Y'):
                    try:
                        parsed_date = datetime.datetime.strptime(self.date, fmt)
                        self.date = parsed_date.strftime('%Y-%m-%d')
                        break
                    except ValueError:
                        continue
            except ValueError:
                logger.error(f"Invalid date format: {self.date}")
                raise ValueError(f"Invalid date format: {self.date}")
        
        # Format time to be consistent (HH:MM)
        if isinstance(self.time_of_journey, str):
            try:
                # Try to parse various time formats
                for fmt in ('%H:%M', '%I:%M %p', '%H.%M'):
                    try:
                        parsed_time = datetime.datetime.strptime(self.time_of_journey, fmt)
                        self.time_of_journey = parsed_time.strftime('%H:%M')
                        break
                    except ValueError:
                        continue
            except ValueError:
                logger.error(f"Invalid time format: {self.time_of_journey}")
                raise ValueError(f"Invalid time format: {self.time_of_journey}")
    
    def setup_driver(self):
        """Set up Selenium WebDriver with appropriate options."""
        chrome_options = Options()
        # Run in headless mode
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--disable-gpu")
        # Add user agent to avoid detection
        chrome_options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
        
        # Anti-bot detection measures
        chrome_options.add_argument("--disable-blink-features=AutomationControlled")
        chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
        chrome_options.add_experimental_option('useAutomationExtension', False)
        
        # Initialize browser
        service = Service(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service, options=chrome_options)
        
        # Additional anti-detection measures
        driver.execute_cdp_cmd('Network.setUserAgentOverride', {
            "userAgent": 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        })
        driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
        
        # Set window size to a reasonable desktop size
        driver.set_window_size(1366, 768)
        
        return driver
    
    def extract_price(self, price_text):
        """Extract numeric price from a string"""
        try:
            # Extract digits and decimal point
            price_match = re.search(r'(\d+\.\d+|\d+)', price_text)
            if price_match:
                return float(price_match.group(1))
            return None
        except Exception as e:
            logger.error(f"Error extracting price from '{price_text}': {e}")
            return None
    
    def update_cheapest_ticket(self, price, link):
        """Update the cheapest ticket if a lower price is found."""
        if price and price < self.cheapest_ticket["price"]:
            self.cheapest_ticket["price"] = price
            self.cheapest_ticket["link"] = link
            logger.info(f"Found cheaper ticket: £{price:.2f} at {link}")
    
    def search(self):
        """Search for tickets. To be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement this method.")
    
    def get_result(self):
        """Return the search result."""
        if self.cheapest_ticket["price"] != float('inf'):
            return {
                "price": f"£{self.cheapest_ticket['price']:.2f}",
                "link": self.cheapest_ticket["link"]
            }
        return {"price": "No tickets found", "link": None}


class GreaterAngliaScraper(TrainTicketScraper):
    """Scraper for Greater Anglia website."""
    
    def __init__(self, origin, destination, date, time_of_journey, railcard=False):
        super().__init__(origin, destination, date, time_of_journey, railcard)
        self.name = "Greater Anglia"
        self.base_url = "https://www.buytickets.greateranglia.co.uk"
    
    def search(self):
        """Search for tickets on Greater Anglia website."""
        logger.info(f"Searching for tickets on {self.name}...")
        
        # Format date for URL (DD-MM-YYYY)
        date_obj = datetime.datetime.strptime(self.date, '%Y-%m-%d')
        formatted_date = date_obj.strftime('%d-%m-%Y')
        
        # Driver setup
        driver = self.setup_driver()
        
        try:
            # Navigate to the search page
            driver.get(self.base_url)
            logger.info(f"Opened {self.name} website")
            
            # Handle cookie consent banner
            try:
                # Wait for the cookie dialog to appear
                WebDriverWait(driver, 8).until(
                    EC.presence_of_element_located((By.ID, "CybotCookiebotDialog"))
                )
                
                logger.info("Found CybotCookiebot dialog")
                
                # Try to click the accept button using various selectors
                selectors = [
                    (By.ID, "CybotCookiebotDialogBodyLevelButtonLevelOptinAllowAll"),
                    (By.CLASS_NAME, "CybotCookiebotDialogBodyButton"),
                    (By.CSS_SELECTOR, "button[id='CybotCookiebotDialogBodyLevelButtonLevelOptinAllowAll']"),
                    (By.XPATH, "//button[contains(text(), 'Allow all cookies')]")
                ]
                
                for selector_type, selector_value in selectors:
                    try:
                        allow_button = WebDriverWait(driver, 3).until(
                            EC.element_to_be_clickable((selector_type, selector_value))
                        )
                        allow_button.click()
                        logger.info(f"Clicked cookie button using {selector_type}: {selector_value}")
                        break
                    except (TimeoutException, NoSuchElementException):
                        continue
                
                # Wait for cookie banner to disappear
                time.sleep(2)
                
            except TimeoutException:
                logger.info("No cookie dialog found or it disappeared quickly")
            
            # Wait for the page to fully load and stabilize
            time.sleep(3)
            
            # Take a screenshot for debugging
            driver.save_screenshot("greater_anglia_initial_page.png")
            
            # Wait for search form to load - look for origin input field
            try:
                WebDriverWait(driver, 10).until(
                    EC.presence_of_element_located((By.ID, "jsf-origin-input"))
                )
                logger.info("Search form loaded successfully")
            except TimeoutException:
                logger.error("Timed out waiting for search form to load")
                driver.save_screenshot("greater_anglia_form_not_found.png")
                driver.quit()
                return
            
            # Enter origin - using JavaScript to set the value directly
            try:
                origin_input = driver.find_element(By.ID, "jsf-origin-input")
                
                # First try direct typing
                driver.execute_script("arguments[0].value = '';", origin_input)
                origin_input.send_keys(self.origin)
                logger.info(f"Entered origin: {self.origin}")
                time.sleep(2)  # Wait for autocomplete
                
                # Try to click on autocomplete options
                try:
                    # Wait for dropdown to appear
                    WebDriverWait(driver, 5).until(
                        EC.presence_of_element_located((By.CSS_SELECTOR, "div[role='option']"))
                    )
                    
                    # Select the first option
                    origin_options = driver.find_elements(By.CSS_SELECTOR, "div[role='option']")
                    if origin_options:
                        driver.execute_script("arguments[0].click();", origin_options[0])
                        logger.info(f"Selected origin from dropdown: {self.origin}")
                    else:
                        # If no options found, try pressing Enter
                        origin_input.send_keys(webdriver.Keys.ENTER)
                        logger.warning(f"No origin options found for {self.origin}, pressed Enter")
                except (TimeoutException, NoSuchElementException) as e:
                    logger.warning(f"Could not find origin station dropdown for: {self.origin}, error: {str(e)}")
                    # Try to continue anyway
            except Exception as e:
                logger.error(f"Error entering origin station: {str(e)}")
                driver.save_screenshot("greater_anglia_origin_error.png")
                
            time.sleep(1)  # Short pause before entering destination
            
            # Enter destination - using JavaScript to set the value directly
            try:
                destination_input = driver.find_element(By.ID, "jsf-destination-input")
                
                # First try direct typing
                driver.execute_script("arguments[0].value = '';", destination_input)
                destination_input.send_keys(self.destination)
                logger.info(f"Entered destination: {self.destination}")
                time.sleep(2)  # Wait for autocomplete
                
                # Try to click on autocomplete options
                try:
                    # Wait for dropdown to appear
                    WebDriverWait(driver, 5).until(
                        EC.presence_of_element_located((By.CSS_SELECTOR, "div[role='option']"))
                    )
                    
                    # Select the first option
                    destination_options = driver.find_elements(By.CSS_SELECTOR, "div[role='option']")
                    if destination_options:
                        driver.execute_script("arguments[0].click();", destination_options[0])
                        logger.info(f"Selected destination from dropdown: {self.destination}")
                    else:
                        # If no options found, try pressing Enter
                        destination_input.send_keys(webdriver.Keys.ENTER)
                        logger.warning(f"No destination options found for {self.destination}, pressed Enter")
                except (TimeoutException, NoSuchElementException) as e:
                    logger.warning(f"Could not find destination station dropdown for: {self.destination}, error: {str(e)}")
                    # Try to continue anyway
            except Exception as e:
                logger.error(f"Error entering destination station: {str(e)}")
                driver.save_screenshot("greater_anglia_destination_error.png")
            
            # Select One-way journey type (should be default, but let's make sure)
            try:
                # Look for the one-way/single option using various possible selectors
                one_way_selectors = [
                    (By.ID, "single"),
                    (By.XPATH, "//input[@type='radio' and @id='single']"),
                    (By.XPATH, "//div[@role='option']//input[@type='radio' and @id='single']")
                ]
                
                for selector_type, selector_value in one_way_selectors:
                    try:
                        one_way_option = driver.find_element(selector_type, selector_value)
                        if not one_way_option.is_selected():
                            driver.execute_script("arguments[0].click();", one_way_option)
                            logger.info("Selected one-way journey")
                        break
                    except (NoSuchElementException, ElementNotInteractableException):
                        continue
            except Exception as e:
                logger.warning(f"Could not select one-way journey option: {str(e)}")
            
            # Take a screenshot after filling origin/destination
            driver.save_screenshot("greater_anglia_form_filled.png")
            
            # Set date and time
            try:
                # Find the date input
                date_time_field = WebDriverWait(driver, 5).until(
                    EC.element_to_be_clickable((By.ID, "jsf-outbound-time-input-toggle"))
                )
                
                # Format date for the UI format "Day D Mon • HH:00"
                day_of_week = date_obj.strftime('%a')  # e.g., "Thu" 
                day = str(date_obj.day)  # Day without leading zero
                month_abbr = date_obj.strftime('%b')  # e.g., "May"
                hour = self.time_of_journey.split(':')[0].zfill(2)  # Hour with leading zero
                
                # Format like "Wed 8 May • 09:00"
                formatted_datetime = f"{day_of_week} {day} {month_abbr} • {hour}:00"
                logger.info(f"Attempting to set date/time to: {formatted_datetime}")
                
                # First, try to update the field using JavaScript
                try:
                    driver.execute_script(f"arguments[0].value = '{formatted_datetime}';", date_time_field)
                    logger.info("Used JavaScript to set initial date/time value")
                    
                    # Click on the field to open date picker
                    driver.execute_script("arguments[0].click();", date_time_field)
                    logger.info("Clicked on date/time field")
                    
                    # Wait for date picker to appear
                    WebDriverWait(driver, 5).until(
                        EC.presence_of_element_located((By.CSS_SELECTOR, "div[role='dialog']"))
                    )
                    
                    # Take a screenshot of date picker
                    driver.save_screenshot("greater_anglia_date_picker.png")
                    
                    # Look for the submit/done/select button in the date picker
                    submit_button_selectors = [
                        (By.XPATH, "//button[contains(text(), 'Select date & time')]"),
                        (By.XPATH, "//button[contains(text(), 'Done')]"),
                        (By.CSS_SELECTOR, "button.primary")
                    ]
                    
                    for selector_type, selector_value in submit_button_selectors:
                        try:
                            submit_button = WebDriverWait(driver, 3).until(
                                EC.element_to_be_clickable((selector_type, selector_value))
                            )
                            driver.execute_script("arguments[0].click();", submit_button)
                            logger.info(f"Clicked date picker submit button using {selector_type}: {selector_value}")
                            break
                        except (TimeoutException, NoSuchElementException, ElementNotInteractableException):
                            continue
                    
                    # If submit button not found, try to click outside to close date picker
                    driver.find_element(By.TAG_NAME, "body").click()
                    
                except (NoSuchElementException, ElementNotInteractableException, StaleElementReferenceException) as e:
                    logger.warning(f"Could not interact with date picker UI: {str(e)}")
                    
                    # Fallback: Try to set the field directly via JavaScript
                    script = f"""
                    var dateField = document.getElementById('jsf-outbound-time-input-toggle');
                    if (dateField) {{
                        dateField.value = '{formatted_datetime}';
                        var event = new Event('change');
                        dateField.dispatchEvent(event);
                    }}
                    """
                    driver.execute_script(script)
                    logger.info("Used fallback JavaScript to set date/time")
                    
                    # Attempt to click outside to ensure any popovers are closed
                    driver.find_element(By.TAG_NAME, "body").click()
            
            except (TimeoutException, NoSuchElementException, ElementNotInteractableException) as e:
                logger.error(f"Failed to set date and time: {str(e)}")
                driver.save_screenshot("greater_anglia_date_time_error.png")
                # Try to continue anyway
            
            # Add railcard if needed
            if self.railcard:
                try:
                    # First check if we need to open a section for railcards
                    try:
                        flexi_tab = driver.find_element(By.ID, "flexi-and-seasons-toggle")
                        driver.execute_script("arguments[0].click();", flexi_tab)
                        logger.info("Clicked on Flexi & Seasons tab")
                        time.sleep(1)
                    except NoSuchElementException:
                        logger.info("Flexi & Seasons tab not found, looking for railcard directly")
                    
                    # Look for railcard options 
                    railcard_triggers = [
                        (By.CSS_SELECTOR, "button[data-testid='jsf-promo']"),  # Try promo/railcard button
                        (By.XPATH, "//span[contains(text(), 'Add railcards')]"),  # Look for "Add railcards" text
                        (By.XPATH, "//button[contains(text(), 'railcard')]"),  # Look for any button with "railcard" text
                        (By.CSS_SELECTOR, "div.jBbD2Rs9pM1LSjnksLrl")  # Try the passengers section which includes railcards
                    ]
                    
                    for selector_type, selector_value in railcard_triggers:
                        try:
                            railcard_button = driver.find_element(selector_type, selector_value)
                            driver.execute_script("arguments[0].click();", railcard_button)
                            logger.info(f"Clicked railcard button using {selector_type}: {selector_value}")
                            
                            # Wait for railcard dialog or overlay
                            WebDriverWait(driver, 5).until(
                                EC.presence_of_element_located((By.CSS_SELECTOR, "div[role='dialog'], div[id^='jsf-overlay']"))
                            )
                            
                            # Take screenshot of railcard dialog
                            driver.save_screenshot("greater_anglia_railcard_dialog.png")
                            
                            # Look for railcard options in the dialog
                            railcard_option_selectors = [
                                (By.CSS_SELECTOR, "li[role='option']"),
                                (By.XPATH, "//div[contains(text(), 'Railcard')]"),
                                (By.XPATH, "//span[contains(text(), 'Railcard')]")
                            ]
                            
                            for option_type, option_value in railcard_option_selectors:
                                try:
                                    railcard_options = driver.find_elements(option_type, option_value)
                                    if railcard_options:
                                        driver.execute_script("arguments[0].click();", railcard_options[0])
                                        logger.info("Selected first railcard option")
                                        
                                        # Look for a "Done" or "Add" button to confirm
                                        confirm_buttons = driver.find_elements(
                                            By.XPATH, 
                                            "//button[contains(text(), 'Done') or contains(text(), 'Add') or contains(text(), 'OK')]"
                                        )
                                        if confirm_buttons:
                                            driver.execute_script("arguments[0].click();", confirm_buttons[0])
                                            logger.info("Confirmed railcard selection")
                                        break
                                except (NoSuchElementException, ElementNotInteractableException):
                                    continue
                            break
                        except (TimeoutException, NoSuchElementException):
                            continue
                
                except Exception as e:
                    logger.warning(f"Could not add railcard: {str(e)}")
            
            # Take screenshot before submitting search
            driver.save_screenshot("greater_anglia_before_search.png")
            
            # Submit search
            try:
                search_button_selectors = [
                    (By.CSS_SELECTOR, "button[data-testid='jsf-submit']"),
                    (By.XPATH, "//button[contains(text(), 'Get cheapest tickets')]"),
                    (By.XPATH, "//button[@type='submit']")
                ]
                
                for selector_type, selector_value in search_button_selectors:
                    try:
                        search_button = WebDriverWait(driver, 3).until(
                            EC.element_to_be_clickable((selector_type, selector_value))
                        )
                        driver.execute_script("arguments[0].click();", search_button)
                        logger.info(f"Clicked search button using {selector_type}: {selector_value}")
                        break
                    except (TimeoutException, NoSuchElementException, ElementNotInteractableException):
                        continue
                else:
                    logger.error("Could not find any search button")
                    driver.save_screenshot("greater_anglia_no_search_button.png")
                    driver.quit()
                    return
            
            except Exception as e:
                logger.error(f"Error clicking search button: {str(e)}")
                driver.save_screenshot("greater_anglia_search_button_error.png")
                driver.quit()
                return
            
            # Wait for results to load
            try:
                # Wait for search results to appear - look for common elements that would be on the results page
                selectors = [
                    (By.CSS_SELECTOR, "[data-testid='journey-results']"),
                    (By.CSS_SELECTOR, ".journey-card"),
                    (By.CSS_SELECTOR, ".fare-card"),
                    (By.CSS_SELECTOR, ".fare-price"),
                    (By.CSS_SELECTOR, ".ticket-price")
                ]
                
                for selector_type, selector_value in selectors:
                    try:
                        WebDriverWait(driver, 20).until(
                            EC.presence_of_element_located((selector_type, selector_value))
                        )
                        logger.info(f"Results page loaded, found {selector_value}")
                        break
                    except TimeoutException:
                        continue
                else:
                    logger.warning(f"{self.name}: No results found or page took too long to load")
                    driver.save_screenshot("greater_anglia_no_results.png")
                    # Save the HTML for debugging
                    with open("greater_anglia_page.html", "w", encoding="utf-8") as f:
                        f.write(driver.page_source)
                    logger.info("Saved page HTML for debugging")
                    driver.quit()
                    return
                
                # Let the page fully render
                time.sleep(3)
                
                # Take a screenshot for debugging
                driver.save_screenshot("greater_anglia_results.png")
                
                # Extract prices - try multiple selectors to find prices
                price_selectors = [
                    ".fare-price", 
                    "[data-testid='fare-price']",
                    ".ticket-price",
                    ".journey-price",
                    ".price",
                    "span[data-testid='journey-fare-price']",
                    ".JourneyCard-fareContainer .fare-price"
                ]
                
                found_prices = False
                
                for selector in price_selectors:
                    price_elements = driver.find_elements(By.CSS_SELECTOR, selector)
                    if price_elements:
                        logger.info(f"Found {len(price_elements)} price elements with selector: {selector}")
                        found_prices = True
                        
                        # Get the current URL for constructing links
                        results_url = driver.current_url
                        
                        for price_element in price_elements:
                            try:
                                price_text = price_element.text.strip()
                                logger.info(f"Found price text: {price_text}")
                                price = self.extract_price(price_text)
                                if price:
                                    self.update_cheapest_ticket(price, results_url)
                            except (StaleElementReferenceException, ElementNotInteractableException) as e:
                                logger.warning(f"Error extracting price: {str(e)}")
                        
                        # If we found prices, we can break the loop
                        break
                
                if not found_prices:
                    logger.warning("No price elements found on the results page")
                    
                    # If no prices were found, let's log the page source for debugging
                    with open("greater_anglia_page_source.html", "w", encoding="utf-8") as f:
                        f.write(driver.page_source)
                    logger.info("Saved page source to greater_anglia_page_source.html")
                
                logger.info(f"{self.name} search completed")
            
            except Exception as e:
                logger.error(f"Error parsing results for {self.name}: {str(e)}")
                # Try to save screenshot for debugging
                try:
                    driver.save_screenshot("greater_anglia_error.png")
                    logger.info("Saved screenshot greater_anglia_error.png")
                except Exception:
                    pass
        except Exception as e:
            logger.error(f"Error searching {self.name}: {str(e)}")
            # Try to save screenshot for debugging
            try:
                driver.save_screenshot("greater_anglia_error.png")
                logger.info("Saved screenshot greater_anglia_error.png")
            except Exception:
                pass
        finally:
            driver.quit()


class NationalRailScraper(TrainTicketScraper):
    """Scraper for National Rail website."""
    
    def __init__(self, origin, destination, date, time_of_journey, railcard=False):
        super().__init__(origin, destination, date, time_of_journey, railcard)
        self.name = "National Rail"
        self.base_url = "https://www.nationalrail.co.uk/journey-planner/"
    
    def search(self):
        """Search for tickets on National Rail website."""
        logger.info(f"Searching for tickets on {self.name}...")
        
        # Format date for National Rail (YYYY-MM-DD)
        date_obj = datetime.datetime.strptime(self.date, '%Y-%m-%d')
        
        # Driver setup
        driver = self.setup_driver()
        
        try:
            # Navigate to the search page
            driver.get(self.base_url)
            logger.info(f"Opened {self.name} website")
            
            # Wait for page to load
            WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "input[placeholder='Departing from']"))
            )
            
            # Enter origin
            origin_input = driver.find_element(By.CSS_SELECTOR, "input[placeholder='Departing from']")
            origin_input.clear()
            origin_input.send_keys(self.origin)
            time.sleep(2)  # Wait for autocomplete
            
            # Select first option from dropdown
            try:
                origin_options = WebDriverWait(driver, 5).until(
                    EC.presence_of_all_elements_located((By.CSS_SELECTOR, "li.MuiAutocomplete-option"))
                )
                if origin_options:
                    origin_options[0].click()
            except TimeoutException:
                logger.warning(f"Could not find origin station: {self.origin}")
                driver.quit()
                return
            
            # Enter destination
            destination_input = driver.find_element(By.CSS_SELECTOR, "input[placeholder='Going to']")
            destination_input.clear()
            destination_input.send_keys(self.destination)
            time.sleep(2)  # Wait for autocomplete
            
            # Select first option from dropdown
            try:
                destination_options = WebDriverWait(driver, 5).until(
                    EC.presence_of_all_elements_located((By.CSS_SELECTOR, "li.MuiAutocomplete-option"))
                )
                if destination_options:
                    destination_options[0].click()
            except TimeoutException:
                logger.warning(f"Could not find destination station: {self.destination}")
                driver.quit()
                return
            
            # Find and click the date input to open the date picker
            date_input = WebDriverWait(driver, 10).until(
                EC.element_to_be_clickable((By.XPATH, "//input[contains(@aria-label, 'Date')]"))
            )
            date_input.click()
            
            # Find and click the specific date in the calendar
            day = date_obj.day
            month_year = date_obj.strftime("%B %Y")
            
            # Find the correct month/year view
            calendar_header = WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.XPATH, "//div[contains(@class, 'MuiPickersCalendarHeader-label')]"))
            )
            
            # If the current month/year doesn't match, navigate to the right month
            # Click next month button until we find the correct month
            while month_year not in calendar_header.text:
                next_month_button = driver.find_element(By.XPATH, "//button[contains(@aria-label, 'Next month')]")
                next_month_button.click()
                time.sleep(0.5)
                calendar_header = driver.find_element(By.XPATH, "//div[contains(@class, 'MuiPickersCalendarHeader-label')]")
            
            # Click on the day
            day_button = WebDriverWait(driver, 10).until(
                EC.element_to_be_clickable((By.XPATH, f"//button[contains(@class, 'MuiPickersDay') and not(contains(@class, 'Mui-disabled')) and .//p[text()='{day}']]"))
            )
            day_button.click()
            
            # Set time
            time_input = WebDriverWait(driver, 10).until(
                EC.element_to_be_clickable((By.XPATH, "//input[contains(@aria-label, 'Time')]"))
            )
            time_input.clear()
            time_input.send_keys(self.time_of_journey)
            
            # Click search button
            search_button = WebDriverWait(driver, 10).until(
                EC.element_to_be_clickable((By.XPATH, "//button[contains(@data-test, 'journey-planner-submit')]"))
            )
            search_button.click()
            
            # Wait for results to load
            try:
                WebDriverWait(driver, 20).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, ".journey-option"))
                )
            except TimeoutException:
                logger.warning(f"{self.name}: No results found or page took too long to load")
                driver.quit()
                return
            
            # Extract prices
            price_elements = driver.find_elements(By.CSS_SELECTOR, ".ticket-option-price")
            
            # Get the current URL for constructing links
            results_url = driver.current_url
            
            for price_element in price_elements:
                price_text = price_element.text.strip()
                price = self.extract_price(price_text)
                if price:
                    self.update_cheapest_ticket(price, results_url)
            
            logger.info(f"{self.name} search completed")
            
        except Exception as e:
            logger.error(f"Error searching {self.name}: {str(e)}")
        finally:
            driver.quit()


class TrainlineScraper(TrainTicketScraper):
    """Scraper for The Trainline website."""
    
    def __init__(self, origin, destination, date, time_of_journey, railcard=False):
        super().__init__(origin, destination, date, time_of_journey, railcard)
        self.name = "The Trainline"
        self.base_url = "https://www.thetrainline.com"
    
    def search(self):
        """Search for tickets on The Trainline website."""
        logger.info(f"Searching for tickets on {self.name}...")
        
        # Format date for URL (YYYY-MM-DD)
        date_obj = datetime.datetime.strptime(self.date, '%Y-%m-%d')
        formatted_date = date_obj.strftime('%Y-%m-%d')
        
        # Driver setup
        driver = self.setup_driver()
        
        try:
            # Navigate to the search page
            driver.get(f"{self.base_url}/journey-planner")
            logger.info(f"Opened {self.name} website")
            
            # Accept cookies if the banner appears
            try:
                cookie_accept = WebDriverWait(driver, 5).until(
                    EC.element_to_be_clickable((By.ID, "onetrust-accept-btn-handler"))
                )
                cookie_accept.click()
            except TimeoutException:
                logger.info("No cookie banner found or it disappeared")
            
            # Wait for page to load
            WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "input[data-test='from-location']"))
            )
            
            # Enter origin
            origin_input = driver.find_element(By.CSS_SELECTOR, "input[data-test='from-location']")
            origin_input.clear()
            origin_input.send_keys(self.origin)
            time.sleep(2)  # Wait for autocomplete
            
            # Select first option from dropdown
            try:
                origin_options = WebDriverWait(driver, 5).until(
                    EC.presence_of_all_elements_located((By.CSS_SELECTOR, "li[role='option']"))
                )
                if origin_options:
                    origin_options[0].click()
            except TimeoutException:
                logger.warning(f"Could not find origin station: {self.origin}")
                driver.quit()
                return
            
            # Enter destination
            destination_input = driver.find_element(By.CSS_SELECTOR, "input[data-test='to-location']")
            destination_input.clear()
            destination_input.send_keys(self.destination)
            time.sleep(2)  # Wait for autocomplete
            
            # Select first option from dropdown
            try:
                destination_options = WebDriverWait(driver, 5).until(
                    EC.presence_of_all_elements_located((By.CSS_SELECTOR, "li[role='option']"))
                )
                if destination_options:
                    destination_options[0].click()
            except TimeoutException:
                logger.warning(f"Could not find destination station: {self.destination}")
                driver.quit()
                return
            
            # Set date
            date_input = driver.find_element(By.CSS_SELECTOR, "input[data-test='outbound-date']")
            driver.execute_script("arguments[0].value = '';", date_input)
            date_input.send_keys(formatted_date)
            
            # Set time
            time_dropdown = driver.find_element(By.CSS_SELECTOR, "select[data-test='outbound-time']")
            time_dropdown.click()
            
            # Format time for option selection (e.g., "09:00" to "09:00")
            time_parts = self.time_of_journey.split(':')
            if len(time_parts) == 2:
                hour, minute = time_parts
                hour = hour.zfill(2)  # Ensure 2 digits
                formatted_time = f"{hour}:{minute}"
                
                # Find and select the time option
                time_option = driver.find_element(By.XPATH, f"//option[contains(text(), '{formatted_time}')]")
                time_option.click()
            
            # Add railcard if needed
            if self.railcard:
                try:
                    # Click on railcard dropdown
                    railcard_button = WebDriverWait(driver, 5).until(
                        EC.element_to_be_clickable((By.CSS_SELECTOR, "button[data-test='passenger-railcard-btn']"))
                    )
                    railcard_button.click()
                    
                    # Select first railcard type
                    railcard_option = WebDriverWait(driver, 5).until(
                        EC.element_to_be_clickable((By.CSS_SELECTOR, "button[data-test='railcard-option']"))
                    )
                    railcard_option.click()
                    
                    # Click done button
                    done_button = driver.find_element(By.CSS_SELECTOR, "button[data-test='passenger-summary-btn']")
                    done_button.click()
                except (TimeoutException, NoSuchElementException):
                    logger.warning("Could not add railcard")
            
            # Submit search
            search_button = driver.find_element(By.CSS_SELECTOR, "button[data-test='search-button']")
            search_button.click()
            
            # Wait for results to load
            try:
                WebDriverWait(driver, 15).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, "div[data-test='journey-results']"))
                )
            except TimeoutException:
                logger.warning(f"{self.name}: No results found or page took too long to load")
                driver.quit()
                return
            
            # Extract prices
            time.sleep(3)  # Let AJAX finish loading
            price_elements = driver.find_elements(By.CSS_SELECTOR, "span[data-test='fare-price']")
            
            # Get the current URL for constructing links
            results_url = driver.current_url
            
            for price_element in price_elements:
                price_text = price_element.text.strip()
                price = self.extract_price(price_text)
                if price:
                    self.update_cheapest_ticket(price, results_url)
            
            logger.info(f"{self.name} search completed")
            
        except Exception as e:
            logger.error(f"Error searching {self.name}: {str(e)}")
        finally:
            driver.quit()


def search_all_sites(origin, destination, date, time_of_journey, railcard=False):
    """Search all sites and return results."""
    results = {}
    
    # Create scrapers
    scrapers = [
        GreaterAngliaScraper(origin, destination, date, time_of_journey, railcard),
        NationalRailScraper(origin, destination, date, time_of_journey, railcard),
        TrainlineScraper(origin, destination, date, time_of_journey, railcard)
    ]
    
    # Search on each site
    for scraper in scrapers:
        try:
            scraper.search()
            results[scraper.name] = scraper.get_result()
        except Exception as e:
            logger.error(f"Error with {scraper.name} scraper: {str(e)}")
            results[scraper.name] = {"price": "Error", "link": None}
    
    # Find overall cheapest
    cheapest_site = None
    cheapest_price = float('inf')
    
    for site, result in results.items():
        if "price" in result and result["price"] != "No tickets found" and result["price"] != "Error":
            price_value = float(result["price"].replace("£", ""))
            if price_value < cheapest_price:
                cheapest_price = price_value
                cheapest_site = site
    
    if cheapest_site:
        results["cheapest"] = {
            "site": cheapest_site,
            "price": results[cheapest_site]["price"],
            "link": results[cheapest_site]["link"]
        }
    else:
        results["cheapest"] = {
            "site": "None",
            "price": "No tickets found",
            "link": None
        }
    
    return results


def main():
    """Main function to run the script."""
    parser = argparse.ArgumentParser(description="Train Ticket Scraper")
    parser.add_argument("--origin", required=True, help="Origin station")
    parser.add_argument("--destination", required=True, help="Destination station")
    parser.add_argument("--date", required=True, help="Date of journey (YYYY-MM-DD)")
    parser.add_argument("--time", required=True, help="Time of journey (HH:MM)")
    parser.add_argument("--railcard", action="store_true", help="Use railcard discount")
    
    args = parser.parse_args()
    
    results = search_all_sites(
        args.origin,
        args.destination,
        args.date,
        args.time,
        args.railcard
    )
    
    # Print results
    print("\nSearch Results:")
    print("-" * 50)
    
    for site, result in results.items():
        if site != "cheapest":
            print(f"{site}:")
            if "price" in result:
                print(f"  Price: {result['price']}")
            if "link" in result and result["link"]:
                print(f"  Link: {result['link']}")
            print()
    
    print("Cheapest Option:")
    print("-" * 50)
    if results["cheapest"]["site"] != "None":
        print(f"Site: {results['cheapest']['site']}")
        print(f"Price: {results['cheapest']['price']}")
        print(f"Link: {results['cheapest']['link']}")
    else:
        print("No tickets found on any site.")


if __name__ == "__main__":
    main()