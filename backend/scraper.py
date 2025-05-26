from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.keys import Keys
import time
from datetime import datetime
import json

class TicketFinder():
    """
    Train Ticket Scraper

    This class scrapes train ticket information from the National Rail website (nationalrail.co.uk)

    The class takes the following parameters:
    - Origin station
    - Destination station
    - Date of journey (Formate: DD/MM/YYYY)
    - Earliest time of departure (Format: HH:MM) (00:00 if not provided)
    - Latest time of departure (Format: HH:MM) (23:99 if not provided)
    - Railcard(s) (Must be provided in its 3 digit code form (see Railcards.txt) ), treated as a list so that a railcard can be used for each passenger ([] if not provided) 
    - Passengers aged 16+ (1 if not provided) (must be lower than or equal to 9)
    - Passengers aged 5-15 (None if not provided) (must be lower than or equal to 9)

    Getter methods return a list of possible journeys (can be sorted by price or speed or time) as well as a link to the tickets page. 
    (unfortunetly cannot provide links to individual tickets as the individual ticket purchase pages are created using cookies & JavaScript and not urls)
    """

    #To do: 
    # - Allow for returns
    # Step 1: Change __init__ to take in a return date and time and a Type parameter (single or return)
    # Step 2: For returns, set Earliest and latest so that both time frames are covered
    # Step 3: Edit Parse_Journey_Data so that if the journey is a return, it gets the return journey details as well
    # - Make getter methods that return the fastest, cheapest, earliest, latest journeys. as well as methods that return lists of journeys sorted in the same wa
    def __init__(self, Origin, Destination, Date, Type="single", Return_Date=None, Earliest_Outbound=(00,00), Latest_Outbound=(23,59), Earliest_Inbound=(00,00), Latest_Inbound=(23,59), Railcards=[], Adults=1, Children=None):
        
        self.Origin_Station = Origin
        self.Destination_Station = Destination
        self.Date_Of_Journey_Outbound = Date
        self.Earliest_Outbound = Earliest_Outbound
        self.Latest_Outbound = Latest_Outbound
        
        #Values for the return journey
        self.Type = Type
        self.Date_Of_Journey_Inbound = Return_Date
        self.Earliest_Inbound = Earliest_Inbound
        self.Latest_Inbound = Latest_Inbound

        self.Railcards = Railcards
        self.Adult_Passengers = Adults
        self.Child_Passengers = Children

        self.Link_To_Tickets = None
        self.Journeys = []
        self.Journeys_Data_RAW = {}
        self.All_Journeys_Found = False

        #Creates a selenium web driver configuration to run in headless mode (no GUI) with performance logs (allows for identifying request responses) enabled
        self.Web_Driver_Config = webdriver.ChromeOptions()
        self.Web_Driver_Config.add_argument("--headless")
        self.Web_Driver_Config.set_capability('goog:loggingPrefs', {'performance': 'ALL'})

    def Search(self):
        """ Method that calls self.Find_Journeys, once if the journey is a single, twice if it is a return"""
        
        # time.sleep(10)
        # print("cmon mush",flush=True)
        # return [["sigma sigma on the wall"],["boy"]]

        Start_Time = time.time()

        print("Searching for trains from", self.Origin_Station, "to", self.Destination_Station, "on", self.Date_Of_Journey_Outbound)
        self.Date_Of_Journey = self.Date_Of_Journey_Outbound
        self.Earliest_Departure = self.Earliest_Outbound
        self.Latest_Departure = self.Latest_Outbound
        
        self.Journeys_Outbound = self.Find_Journeys()
        if len(self.Journeys_Outbound) > 0:
            self.Link_To_Tickets = self.Journeys_Outbound[0]["Link"]
        
        self.Journeys_Inbound = None
        if self.Type == "return":
            print("Searching for trains from", self.Origin_Station, "to", self.Destination_Station, "on", self.Date_Of_Journey_Inbound)
            
            self.Journeys_Data_RAW = {}
            self.All_Journeys_Found = False
            
            #Sets the earliest and latest times for the return journey
            New_Origin = self.Destination_Station
            New_Destination = self.Origin_Station
            self.Origin_Station = New_Origin
            self.Destination_Station = New_Destination
            
            if self.Date_Of_Journey_Inbound == None:
                self.Date_Of_Journey_Inbound = self.Date_Of_Journey_Outbound

            self.Date_Of_Journey = self.Date_Of_Journey_Inbound
            self.Earliest_Departure = self.Earliest_Inbound
            self.Latest_Departure = self.Latest_Inbound
            
            self.Journeys_Inbound = self.Find_Journeys()
        
        # End_Time = time.time()
        # print("Time taken to find journeys:", End_Time - Start_Time)
        # if self.Journeys_Outbound:
        #     print("\nOutbound Journeys:")
        #     for idx, journey in enumerate(self.Journeys_Outbound, 1):
        #         print(f"Journey {idx}:")
        #         for key, value in journey.items():
        #             print(f"  {key}: {value}")
        #     print("-" * 30)

        # if self.Journeys_Inbound:
        #     print("\nReturn Journeys:")
        #     for idx, journey in enumerate(self.Journeys_Inbound, 1):
        #         print(f"Journey {idx}:")
        #         for key, value in journey.items():
        #             print(f"  {key}: {value}")
        #             # Show single and return price if available
        #             if "Price" in journey:
        #                 print(f"  Single Price: {journey['Price']}")
        #             if "Return_Price" in journey and journey["Return_Price"] is not None:
        #                 print(f"  Return Price: {journey['Return_Price']}")
        #     print("-" * 30)
        return [self.Journeys_Outbound, self.Journeys_Inbound]
        
        
    def Find_Journeys(self):
        """Method that uses a selenium web driver to scrape possible journeys that satisfy the criteria given in __init__ from the nationalrail website"""
        
        #Initializes a chrome web driver
        WebDriver = webdriver.Chrome(options=self.Web_Driver_Config)

        #Navigates to www.nationalrail.co.uk
        WebDriver.get("https://www.nationalrail.co.uk/")

        #Waits until the button used to Accept All cookies appears on the page and then clicks it (returns an empty array if this fails to happen in 20 seconds)
        try:
            Cookies_Button = WebDriverWait(WebDriver,20).until(EC.element_to_be_clickable((By.ID, 'onetrust-accept-btn-handler')))
        except:
            print("ERROR: Unable to locate the accept cookies button")
            return []
        Cookies_Button.click()

        #Waits until the button used to bring up the journey planner window shows up and then clicks it (returns an empty array if this fails to happen in 5 seconds)
        try:
            JP_Button = WebDriverWait(WebDriver,5).until(EC.element_to_be_clickable((By.XPATH, '//*[@aria-label="Plan Your Journey"]')))
            #JP_Button = WebDriverWait(WebDriver,5).until(EC.element_to_be_clickable((By.XPATH, '//*[@data-testid="jp-preview-btn"]')))
        except:
            print("ERROR: Unable to locate the journey planner button")
            return []
        
        WebDriver.execute_script("arguments[0].scrollIntoView({block: 'center'});", JP_Button)
        time.sleep(0.5)
        JP_Button.click()        
        #time.sleep(1)

        #Now that the journey planner menu has appeared, populatge it with the relevant fields and click enter
        if not self.Populate_Journey_Planner(WebDriver):
            print("ERROR: Unable to populate the journey planner menu")
            return []
        
        #Now the available journeys should be displayed
        #Firstly checks if there is a "We couldn't find any services for the journey you have requested. Please check your selection criteria" message
        #If no journeys are found, increase the time by 1 hour and try again
        Journeys_Found = False
        Search_Hour = self.Earliest_Departure[0]
        while not Journeys_Found:
            try:
                No_Journeys_Message = WebDriverWait(WebDriver,5).until(EC.presence_of_element_located((By.XPATH , '//*[contains(text(), "Please check your selection criteria.")]')))
                #time.sleep(1)
                Search_Hour += 1
                Incremented_Time = [Search_Hour, self.Earliest_Departure[1]]
                self.Populate_Earliest_Time(WebDriver, Incremented_Time)
                #Clicks the search button
                Search_Button = WebDriver.find_element(By.ID, "button-jp")  
                Search_Button.click()
                time.sleep(0.5)
            except Exception as e:
                #If the message wasnt detected, also check if any journeys with self.Date_Of_Journey are present
                if self.Check_For_Valid_Journeys(WebDriver):
                    #If they are, set Journeys_Found to True and break out of the loop
                    Journeys_Found = True
                    print("Journeys found")
                #If they arent any valid journeys, click the edit journey and increment the time by 1 hour
                else:
                    try:
                        Edit_Journey_Button = WebDriverWait(WebDriver, 5).until(EC.element_to_be_clickable((By.ID, "button-journey-planner-query")))
                        Edit_Journey_Button.click()
                        time.sleep(1)
                        Search_Hour += 1
                        Incremented_Time = [Search_Hour, self.Earliest_Departure[1]]
                        self.Populate_Earliest_Time(WebDriver, Incremented_Time)
                        #Clicks the search button
                        Search_Button = WebDriver.find_element(By.ID, "button-jp")  
                        Search_Button.click()
                        time.sleep(3)

                    except Exception as e:
                        print(f"ERROR: Unable to locate the 'Edit journey' button: {e}")
                        return []

        #Next load in as many journeys as possible (Click "View more journeys" until maximum journeys reached message appears)
        self.Load_All_Journeys(WebDriver)
        
        #If lookinng for journeys with railcards, prices need to be scraped directly from the page as the json data does not account for railcards, therefore the scraper needs to wait for all the journeys to appear on the page before scraping
        # if self.Railcards != []:
        #     #Identify the number of journey elements on the page
        #     Journeys_On_Page_Old = len(WebDriver.find_elements(By.XPATH, "//section[contains(@data-testid, 'result-card-section-outward')]"))
        #     time.sleep(3)
        #     #Identify the number of journey elements on the page again
        #     Journeys_On_Page_New = len(WebDriver.find_elements(By.XPATH, "//section[contains(@data-testid, 'result-card-section-outward')]"))
            
        #     #Waits until the number of journey elements on the page stops changing 
        #     while Journeys_On_Page_Old != Journeys_On_Page_New:
        #         Journeys_On_Page_Old = len(WebDriver.find_elements(By.XPATH, "//section[contains(@data-testid, 'result-card-section-outward')]"))
        #         time.sleep(3)
        #         #Identify the number of journey elements on the page again
        #         Journeys_On_Page_New = len(WebDriver.find_elements(By.XPATH, "//section[contains(@data-testid, 'result-card-section-outward')]"))
        # else:
        #     #Wait for 3 seconds to allow the journeys to load
        time.sleep(3)
        
        #Now that all of the journeys have been loaded, scrape the relevant details from the json data sent by the server
        Journeys = self.Get_Journeys(WebDriver)
        
        #Next check if self.All_Journeys_Found is True, if it isnt, set the departure time in the journey planner to the latest time and click search again (repeat this process until the last journey found stays the same )
        while not self.All_Journeys_Found:
            #Gets the time of the latest journey found so far
            Latest_Current_Journey_Time = Journeys[-1]["Start_Time"].split(":")
            print("setting time tp", Latest_Current_Journey_Time)
            Edit_Journey_Button = WebDriverWait(WebDriver, 5).until(EC.element_to_be_clickable((By.ID, "button-journey-planner-query")))
            WebDriver.execute_script("arguments[0].scrollIntoView({block: 'center'});", Edit_Journey_Button)
            time.sleep(5)
            Edit_Journey_Button.click()
            self.Populate_Earliest_Time(WebDriver, Latest_Current_Journey_Time)
            time.sleep(0.5)
            #Clicks the search button
            Search_Button = WebDriver.find_element(By.ID, "button-jp")  
            Search_Button.click()

            #Next load in as many journeys as possible (Click "View more journeys" until maximum journeys reached message appears)
            self.Load_All_Journeys(WebDriver)
            
            # #If lookinng for journeys with railcards, prices need to be scraped directly from the page as the json data does not account for railcards, therefore the scraper needs to wait for all the journeys to appear on the page before scraping
            # if self.Railcards != []:
            #     #Identify the number of journey elements on the page
            #     Journeys_On_Page_Old = len(WebDriver.find_elements(By.XPATH, "//section[contains(@data-testid, 'result-card-section-outward')]"))
            #     time.sleep(3)
            #     #Identify the number of journey elements on the page again
            #     Journeys_On_Page_New = len(WebDriver.find_elements(By.XPATH, "//section[contains(@data-testid, 'result-card-section-outward')]"))
                
            #     #Waits until the number of journey elements on the page stops changing 
            #     while Journeys_On_Page_Old != Journeys_On_Page_New:
            #         Journeys_On_Page_Old = len(WebDriver.find_elements(By.XPATH, "//section[contains(@data-testid, 'result-card-section-outward')]"))
            #         time.sleep(3)
            #         #Identify the number of journey elements on the page again
            #         Journeys_On_Page_New = len(WebDriver.find_elements(By.XPATH, "//section[contains(@data-testid, 'result-card-section-outward')]"))
            # else:
            #     #Wait for 3 seconds to allow the journeys to load
            time.sleep(3)
            
            #Now that all of the journeys have been loaded, scrape the relevant details from the json data sent by the server
            New_Journeys = self.Get_Journeys(WebDriver)
            #If the last journey found is the same as the last journey found before, set self.All_Journeys_Found to True
            if len(New_Journeys) > 0:
                print(New_Journeys[-1]["Start_Time"], Journeys[-1]["Start_Time"])
                if [New_Journeys[-1]["Start_Time"],New_Journeys[-1]["Arrival_Time"]] == [Journeys[-1]["Start_Time"],Journeys[-1]["Arrival_Time"]]:
                    self.All_Journeys_Found = True
                else:
                    #Removes duplicate journeys that may happen if the first of the new journeys is the same as the last of the old journeys
                    print([New_Journeys[0]["Start_Time"],New_Journeys[0]["Arrival_Time"]],[Journeys[-1]["Start_Time"],Journeys[-1]["Arrival_Time"]])
                    if [New_Journeys[0]["Start_Time"],New_Journeys[0]["Arrival_Time"]] == [Journeys[-1]["Start_Time"],Journeys[-1]["Arrival_Time"]]:
                        New_Journeys.pop(0)
                    Journeys.extend(New_Journeys)

        #Iterates through the journeys and removes duplicates.
        Journeys_No_Duplicates = []
        if len(Journeys) > 0:
            Journeys_No_Duplicates.append(Journeys[0])
            for i in range(1, len(Journeys)):
                if Journeys[i]["Start_Time"] != Journeys_No_Duplicates[-1]["Start_Time"] or Journeys[i]["Arrival_Time"] != Journeys_No_Duplicates[-1]["Arrival_Time"]:
                    Journeys_No_Duplicates.append(Journeys[i])
                else:
                    pass
        elif len(Journeys) == 1:
            Journeys_No_Duplicates.append(Journeys[0])
                
        Journeys = Journeys_No_Duplicates

        WebDriver.quit()

        return Journeys

    def Load_All_Journeys(self, WebDriver):        
        """Method that loads all of the journeys from the page by clicking the "View more journeys" button until it is no longer present"""
        Maximum_Journeys_Reached = False
        Latest_Request_ID = None
        while not Maximum_Journeys_Reached:
            try:
                View_More_Button = WebDriverWait(WebDriver,2.5).until(EC.presence_of_element_located((By.XPATH, "//button[@aria-label = 'View later trains']")))
                #time.sleep(1)
                WebDriver.execute_script("arguments[0].scrollIntoView({block: 'center'});", View_More_Button)
                time.sleep(1)
                View_More_Button.click()
                
                #Checks if pressing the button actually loads any more journeys, if not, set Maximum_Journeys_Reached to True
                New_Journeys_Loaded = self.New_Journeys_Loaded(WebDriver, Latest_Request_ID)
                print(New_Journeys_Loaded)
                if not New_Journeys_Loaded[0]:
                    Maximum_Journeys_Reached = True
                else:
                    Latest_Request_ID = New_Journeys_Loaded[1]
            
            #Sets Maximum_Journeys_Reached to True if the button is no longer present
            except Exception as e:
                print(e)
                Maximum_Journeys_Reached = True
    
    def Populate_Journey_Planner(self, WebDriver):
        """Method that populates the journey planner menu with the relevant fields and clicks enter"""
        #Now that the journey planner menu has popped up, get the relevant fields.
        Origin_Field = WebDriver.find_element(By.ID, "jp-origin")
        Destination_Field = WebDriver.find_element(By.ID, "jp-destination")
        Date_Menu = WebDriver.find_element(By.ID, "leaving-date")
        #Railcard_Button = WebDriver.find_element(By.XPATH, '//*[@data-testid="rail-card-button-initial"]')
        Railcard_Button = WebDriver.find_element(By.XPATH, '//*[@aria-label="Add railcard"]')
        Adults_Field = WebDriver.find_element(By.ID, "adults")
        Children_Field = WebDriver.find_element(By.ID, "children")

        #Fills out the origin and destination fields
        Origin_Field.click()
        Origin_Field.send_keys(self.Origin_Station)
        Origin_Field.send_keys(Keys.ENTER)
        Destination_Field.send_keys(self.Destination_Station)


        #Fills out the date field
        if not self.Populate_Date_Picker(WebDriver, Date_Menu, self.Date_Of_Journey):
            return False
        
        #Fills out the earliest time field
        self.Populate_Earliest_Time(WebDriver,self.Earliest_Departure)
        time.sleep(0.25)

        #Fills out the adult and child fields if they require changing
        if self.Adult_Passengers != 1:
            Adults_Field.click()
            Adults_Option = Adults_Field.find_element(By.XPATH, f".//option[@value='{self.Adult_Passengers}']")
            #Adults_Option = WebDriver.find_element(By.XPATH, f"//option[@value='{self.Adult_Passengers}' and contains(@data-testid, 'adults')]")
            Adults_Option.click()
        if self.Child_Passengers != None:
            Children_Field.click()
            Children_Option = Children_Field.find_element(By.XPATH, f".//option[@value='{self.Child_Passengers}']")
            #Children_Option = WebDriver.find_element(By.XPATH, f"//option[@value='{self.Child_Passengers}' and contains(@data-testid, 'children')]")
            Children_Option.click()
        
        #Fills out the railcard field if it requires changing
        if self.Railcards != []:
            #Dictionary to hold where railcards have been added (and how many) so that their "dropdown-select-count" can be incremented if needed
            Added_Railcards = {}
            #Adds the first railcard
            WebDriver.execute_script("arguments[0].scrollIntoView({block: 'center'});", Railcard_Button)
            time.sleep(0.5)
            Railcard_Button.click()
            Railcard_Menu = WebDriver.find_element(By.ID, "railcard-0")
            Railcard_Menu.click()
            Railcard_Option = Railcard_Menu.find_element(By.XPATH, f"//option[@value='{self.Railcards[0]}']") 
            #Railcard_Option = WebDriver.find_element(By.XPATH, f"//option[@value='{self.Railcards[0]}' and contains(@data-testid, 'railcard')]")
            Railcard_Option.click()
            #Loops through any additional railcards and adds them
            if len(self.Railcards) > 1:
                #time.sleep(0.5)
                Added_Railcards[self.Railcards[0]] = {"Counter Dropdown":WebDriver.find_element(By.ID, f"railcard-0-count"), "Count":1, "Added Railcard Number":0}
    
                Railcard_Count = 1
                for Railcard in self.Railcards[1:]:
                    
                    #First checks if the Railcard with the same id has been added before, if it has, increment into "dropdown-select-count", if not, add a new railcard
                    if Railcard in Added_Railcards:
                        WebDriver.execute_script("arguments[0].scrollIntoView({block: 'center'});", Added_Railcards[Railcard]["Counter Dropdown"])
                        time.sleep(0.5)
                        Added_Railcards[Railcard]["Counter Dropdown"].click()
                        Count_Option = Added_Railcards[Railcard]["Counter Dropdown"].find_element(By.XPATH, f".//option[@value='{Added_Railcards[Railcard]['Count']+1}']")
                        #Count_Option = WebDriver.find_element(By.XPATH, f"//option[@value='{Added_Railcards[Railcard]['Count']+1}' and contains(@data-testid, 'railcard-{Added_Railcards[Railcard]['Added Railcard Number']}-count')]")
                        Count_Option.click()
                        Added_Railcards[Railcard]["Count"] += 1
                    else:    
                        #Add_Railcard_Button = WebDriver.find_element(By.XPATH, '//*[@data-testid="rail-card-button-additional"]')
                        Add_Railcard_Button = WebDriver.find_element(By.XPATH, '//*[@arua-label="Add another railcard"]')
                        WebDriver.execute_script("arguments[0].scrollIntoView({block: 'center'});", Add_Railcard_Button)
                        time.sleep(0.5)
                        Add_Railcard_Button.click()
                        Railcard_Menu = WebDriver.find_element(By.ID, f"railcard-{Railcard_Count}")
                        Railcard_Menu.click()
                        Railcard_Option = Railcard_Menu.find_element(By.XPATH, f"//option[@value='{Railcard}']")
                        #Railcard_Option = WebDriver.find_element(By.XPATH, f"//option[@value='{Railcard}' and contains(@data-testid, 'railcard-{Railcard_Count}')]")
                        Railcard_Option.click()
                        Added_Railcards[Railcard] = {"Counter Dropdown":WebDriver.find_element(By.ID, f"railcard-{Railcard_Count}-count"), "Count":1, "Added Railcard Number":Railcard_Count}
                        Railcard_Count += 1

        #Clicks the search button
        Search_Button = WebDriver.find_element(By.ID, "button-jp")  
        Search_Button.click()

        return True

    def Populate_Date_Picker(self, WebDriver,Date_Menu, Date):
        """Method that populates the date picker menu with a date given in the format DD/MM/YYYY"""
        #The date picker works in the format day month, so any 0s in the day must be removed and the month must be converted to its word equivilant
        Months = ["January", "Febuary", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
        Date = Date.split("/")
        Day = Date[0].lstrip("0")
        Month = Months[int(Date[1])-1]

        #Checks the date already in the date picker and if it is not the same as the one given, clicks the date picker to open it.
        Date_In_Menu = Date_Menu.get_attribute("value").split(" ")
        if Day != Date_In_Menu[0] or Month != Date_In_Menu[1]:

            #Opens the date picker and picks the right date
            Date_Menu.click()

            #Gets the two months that are displayed in the date picker
            Month_Found = False
            while not Month_Found:
                Displayed_Months = WebDriver.find_elements(By.CLASS_NAME, "react-datepicker__current-month")
                print(Displayed_Months[0].text, Displayed_Months[1].text)
                if Month not in Displayed_Months[0].text and Month not in Displayed_Months[1].text:
                    #If the month is not displayed, click the next month button until it is.
                    Next_Month_Button = WebDriver.find_element(By.XPATH, '//*[@aria-label="Next Month"]')
                    try:
                        Next_Month_Button.click()
                        time.sleep(0.5)
                    except:
                        print("ERROR: Month given not found in date picker (probably too far in the future)")
                        return False
                else:
                    Month_Found = True

            #Clicks the right day in the date picker
            Day_Element = WebDriver.find_element(By.XPATH, f"//div[contains(@aria-label, '{Day} {Month}') and not(contains(@aria-label, 'Not available')) and not(contains(@class, 'outside-month'))]")
            time.sleep(0.5)
            Day_Element.click()
        return True

    def Populate_Earliest_Time(self, WebDriver, Time):        
            #Locates the earliest time fields
            Earliest_Time_Fields = [WebDriver.find_element(By.ID, "leaving-hour"), WebDriver.find_element(By.ID, "leaving-min")]
            
            #Fills out the earliest time fields
            Earliest_Time_Fields[0].click() #Hour field
            Earliest_Time_Hour_Str = str(Time[0]).zfill(2) #Converts the hour to a string and adds a 0 if needed
            #Earlies_Time_Minute_Str = str(Time[1]).zfill(2) #Converts the minute to a string and adds a 0 if needed
            Earliest_Time_Hour_Option = Earliest_Time_Fields[0].find_element(By.XPATH, f".//option[@value='{Earliest_Time_Hour_Str}']")
            #Earliest_Time_Hour_Option = WebDriver.find_element(By.XPATH, f"//option[@value='{Earliest_Time_Hour_Str}' and contains(@data-testid, 'leaving-hour')]")
            Earliest_Time_Hour_Option.click()
            Earliest_Time_Fields[1].click() #Minute field (Sets to 00 as the minute checking will be done later)
            Earliest_Time_Minute_Option = Earliest_Time_Fields[1].find_element(By.XPATH, ".//option[@value='00']")
            #Earliest_Time_Minute_Option = WebDriver.find_element(By.XPATH, f"//option[@value='00' and contains(@data-testid, 'leaving-min')]")
            Earliest_Time_Minute_Option.click()

    def New_Journeys_Loaded(self, WebDriver, Latest_Request_ID):
        """Method that checks if any new journeys have been loaded after clicking the "View more" button by seeing if any new "Network.responseReceived" requests from the journey-planner endpoint have been recieived
           If new journeys have been loaded, return True and add the raw journey data to self.Journeys_Data_RAW"""
        time.sleep(1)
        Network_Logs = WebDriver.get_log("performance")
        #Iterates through the logs and finds the responses to the get journeys requests
        Request_IDs = []
        for log in Network_Logs:
            log_Data = json.loads(log["message"])["message"]
            if log_Data["method"] == "Network.responseReceived":
                if log_Data["params"]["response"]["url"] == "https://jpservices.nationalrail.co.uk/journey-planner":
                    #If the request response was from the right url, get the request id
                    Request_IDs.append(log_Data["params"]["requestId"])

        print(len(Request_IDs))
        #Compares the last requestId to Latest_Request_ID, if they are the same, no new journeys have been loaded
        if len(Request_IDs) > 0:
            if Latest_Request_ID:
                if Request_IDs[-1] == Latest_Request_ID:
                    #If they are the same, no new journeys have been loaded, return False
                    return [False, Latest_Request_ID]
                else:
                    #If they are different, new journeys have been loaded, set Latest_Request_ID to the last one, add the RAW journey data from the GET request to self.Journeys_Data_RAW and return True
                    Journey_Data = WebDriver.execute_cdp_cmd("Network.getResponseBody", {"requestId": Request_IDs[-1]})
                    Journey_Data_JSON = json.loads(Journey_Data["body"])
                    self.Journeys_Data_RAW[Request_IDs[-1]] = Journey_Data_JSON
                    
                    Latest_Request_ID = Request_IDs[-1]
                    return [True, Latest_Request_ID]
           
            #if Latest_Request_ID hasnt been set yet, set it to the last request ID
            else:
                Journey_Data = WebDriver.execute_cdp_cmd("Network.getResponseBody", {"requestId": Request_IDs[-1]})
                Journey_Data_JSON = json.loads(Journey_Data["body"])
                self.Journeys_Data_RAW[Request_IDs[-1]] = Journey_Data_JSON
                Latest_Request_ID = Request_IDs[-1]
                return [True, Latest_Request_ID]
            
        #If no request IDs were found, return False
        self.All_Journeys_Found = True
        return [False, Latest_Request_ID]

    def Get_Journeys(self, WebDriver,Reset=True):
        """Method that scrapes journey details from the json data sent by the server"""

        Journeys = []
        #Iterates through self.Journeys_Data_RAW and parses the data of each journey
        for Request_ID in self.Journeys_Data_RAW:
            #Parses the response data and appends it to the journeys list
            Parsed_Journey_Data = self.Parse_Journey_Data(WebDriver, self.Journeys_Data_RAW[Request_ID])
            Journeys.extend(Parsed_Journey_Data)

        
        #Resets self.Journeys_Data_RAW
        if Reset:
            self.Journeys_Data_RAW = {}
        
        return Journeys

    def Get_Journeys_OLD(self, WebDriver): #REMOVE FUNCTION IF NOT NEEDED
        """Method that scrapes journey details from the json data sent by the server"""

        Journeys = []
        #Firstly gets the network logs from the web driver
        Network_Logs = WebDriver.get_log("performance")

        #Iterates through the logs and finds the responses to the get journeys requests
        Request_IDs = []
        for log in Network_Logs:
            log_Data = json.loads(log["message"])["message"]
            if log_Data["method"] == "Network.responseReceived":
                if log_Data["params"]["response"]["url"] == "https://jpservices.nationalrail.co.uk/journey-planner":
                    #If the request response was from the right url, get the request id
                    Request_IDs.append(log_Data["params"]["requestId"])

        #Iterates through the Request IDs and gets the response data
        for request_id in Request_IDs:
            try:
                Response_Data = WebDriver.execute_cdp_cmd("Network.getResponseBody", {"requestId": request_id})
                Response_Data_JSON = json.loads(Response_Data["body"])
                #Parses the response data and appends it to the journeys list
                Parsed_Journey_Data = self.Parse_Journey_Data(WebDriver, Response_Data_JSON)
                Journeys.extend(Parsed_Journey_Data)
            except Exception as e:
                print(f"Error getting response data: {e}")
                
        return Journeys
    
    def Parse_Journey_Data(self,WebDriver, Response_Data_JSON):
        """Method that takes json data containing the details of multiple (usually 5) journeys and parses it into a list of dictionaries containing the relevant details"""
        Journeys = []
        Link = WebDriver.current_url

        #

        for journey in Response_Data_JSON["outwardJourneys"]:
            
            #Skip the journey if it is cancelled
            if journey["rawStatus"] == "CANCELLED":
                continue
            
            id = journey["id"]
            Start_Time_RAW = journey["timetable"]["scheduled"]["departure"]
            Start_Time = datetime.fromisoformat(Start_Time_RAW).strftime("%H:%M")
            Departure_Date = datetime.fromisoformat(Start_Time_RAW).strftime("%d/%m/%Y")
            Arrival_Time_RAW = journey["timetable"]["scheduled"]["arrival"]
            Arrival_Time = datetime.fromisoformat(Arrival_Time_RAW).strftime("%H:%M")
            Duration = journey["duration"]
            Direct = len(journey["legs"]) == 1
            Changes = len(journey["legs"]) - 1

            # #If a railcard is being used, the price needs to be scraped from the page as the json data does not account for railcards
            # if self.Railcards != []:
            #     #Identify the journey element corasponding to the journey based off of the Start_Time and Arrival_Time
            #     Journey_Elements = WebDriver.find_elements(By.XPATH, "//section[contains(@data-testid, 'result-card-section-outward')]")
            #     for Journey_Element in Journey_Elements:
            #         # Extract departure and arrival times in ISO format from the journey element
            #         Journey_Element_Start_Time = Journey_Element.find_element(By.XPATH, ".//span[@data-testid='card-departs-time-status-jp-compact-summary-origin']/time")
            #         Journey_Element_Arrival_Time = Journey_Element.find_element(By.XPATH, ".//span[@data-testid='card-arrives-time-status-jp-compact-summary-destination']/time")
            #         Journey_Element_Start_Time = Journey_Element_Start_Time.get_attribute("datetime")
            #         Journey_Element_Arrival_Time = Journey_Element_Arrival_Time.get_attribute("datetime")
            #         #If the times match Start_Time and Arrival_Time, get the price
            #         if Journey_Element_Start_Time == Start_Time_RAW and Journey_Element_Arrival_Time == Arrival_Time_RAW:
            #             Price = Journey_Element.find_element(By.XPATH, ".//div[contains(@id, 'result-card-price-outward')]/div/span[2]").text
            # #If no railcard is being used, the price can be scraped from the json data
            # else:
            try:
                Price = min(fare["totalPrice"] for fare in journey["fares"]) #Gets lowest price in fares (in pence)
            except:
                continue
            if self.Type == "return":
                try:
                    Return_Price = min(fare["totalPrice"] for fare in journey["fares"] if fare["direction"] == "RETURN") 
                except:
                    Return_Price = None
            
            #Iterates through the legs of the journey and checks if any of them are rail replacement services, if so, set Rail_Replacement to True
            Rail_Replacement = False
            for leg in journey["legs"]:
                if leg["isReplacementBus"]:
                    Rail_Replacement = True
                    break
            
            Journey_Details = {
                "ID": id,
                "Departure_Date": Departure_Date,
                "Start_Time": Start_Time,
                "Arrival_Time": Arrival_Time,
                "Duration": Duration,
                "Price": Price,
                "Direct": Direct,
                "Changes": Changes,
                "Rail_Replacement": Rail_Replacement,
                "Link": Link
            }
            if self.Type == "return":
                Journey_Details["Return_Price"] = Return_Price
                #Creates a unique ID based off of the Departure time and duration. 
                #Journey_Details["Return ID"] = int(Start_Time.split(":")[0])+int(Start_Time.split(":")[1])+int(Duration.split(":")[0])+int(Duration.split(":")[1]) #Unique ID for the return journey

            #Ignores journeys that are not on the right date
            if Journey_Details["Departure_Date"] == self.Date_Of_Journey:
                
                #If the journey time exceeds self.Latest_Departure, set self.All_Journeys_Found to True and break the loop
                Start_Time_Hour = int(Start_Time.split(":")[0])
                Start_Time_Minute = int(Start_Time.split(":")[1])
                if (Start_Time_Hour, Start_Time_Minute) > self.Latest_Departure:
                    self.All_Journeys_Found = True
                    break
                #If the journey time is before self.Earliest_Departure, dont add it to journeys
                elif (Start_Time_Hour, Start_Time_Minute) < self.Earliest_Departure:
                    continue

                #If the journey is on the right date and within self.Latest_Departure, append it to the list of journeys
                Journeys.append(Journey_Details)
            
            #If the journey is on the day after the right date, break the loop and set self.All_Journeys_Found to True
            else:
                continue
                
                

        return Journeys
    
    def Check_For_Valid_Journeys(self, WebDriver):
        """Method that checks for the presence of journies that depart on self.Departure_Date, returns True if they are found"""
        
        Network_Logs = WebDriver.get_log("performance")
        #Iterates through the logs and finds the responses to the get journeys request (from clicking search on the journey planner)
        Request_IDs = []
        for log in Network_Logs:
            log_Data = json.loads(log["message"])["message"]
            if log_Data["method"] == "Network.responseReceived":
                if log_Data["params"]["response"]["url"] == "https://jpservices.nationalrail.co.uk/journey-planner":
                    #If the request response was from the right url, get the request id
                    Request_IDs.append(log_Data["params"]["requestId"])
        for Request_ID in Request_IDs:
            try:
                Journey_Data = WebDriver.execute_cdp_cmd("Network.getResponseBody", {"requestId": Request_ID})
                Journey_Data_JSON = json.loads(Journey_Data["body"])
                self.Journeys_Data_RAW[Request_ID] = Journey_Data_JSON
            except Exception as e:
                print(f"Error getting response data: {e}")

        #First gets any journeys that have been sent by the server
        Journeys = self.Get_Journeys(WebDriver,Reset=False)
        #Then checks if any of the journeys have a departure date that matches self.Departure_Date
        for Journey in Journeys:
            if Journey["Departure_Date"] == self.Date_Of_Journey:
                #resets self.Journeys_Data_RAW 
                #self.Journeys_Data_RAW = {}
                return True
        #If none found, return False
        #resets self.Journeys_Data_RAW 
        #self.Journeys_Data_RAW = {}
        return False

    def Scrape_Journeys(self, WebDriver): #NOT USED 
        """Method that scrapes all of the journeys from the page and returns their details in a list. Each journey is a dictionary like: {Start_Time, Arrival_Time, Duration, Price, Direct, Rail_Replacement}"""
        Journeys = []
        
        #Gets all of the journey elements
        Journey_Elements = WebDriver.find_elements(By.XPATH, "//section[contains(@data-testid, 'result-card-section-outward')]")
        Journies_Url = WebDriver.current_url
        print(len(Journey_Elements))
        #Iterates through each one and scrapes the relevant details
        for Journey_Element in Journey_Elements:
            #Exception handling to deal with journeys that may be missing any of the details for whatever reason
            try:
                id = Journey_Element.get_attribute("id")
                Start_Time = Journey_Element.find_element(By.XPATH, ".//span[@data-testid='card-departs-time-status-jp-compact-summary-origin']/time").text
                Arrival_Time = Journey_Element.find_element(By.XPATH, ".//span[@data-testid='card-arrives-time-status-jp-compact-summary-destination']/time").text
                Duration = Journey_Element.find_element(By.XPATH, ".//p[contains(@data-testid, 'duration-changes')]/time/span").text
                Price = Journey_Element.find_element(By.XPATH, ".//div[contains(@id, 'result-card-price-outward')]/div/span[2]").text
                Direct = "Direct" in Journey_Element.find_element(By.XPATH, ".//p[contains(@data-testid, 'duration-changes')]/span[2]").text
                Changes = Journey_Element.find_element(By.XPATH, ".//p[contains(@data-testid, 'duration-changes')]/span[2]").text.split(" ")[0]
                try:
                    Journey_Element.find_element(By.XPATH, ".//p[contains(text(), 'Rail Replacement Service')]")
                    Rail_Replacement = True
                except:
                    Rail_Replacement = False

                Journey_Details = {
                    "ID": id,
                    "Start_Time": Start_Time,
                    "Arrival_Time": Arrival_Time,
                    "Duration": Duration,
                    "Price": Price,
                    "Direct": Direct,
                    "Changes": Changes,
                    "Rail_Replacement": Rail_Replacement,
                    "Link": Journies_Url
                }
                Journeys.append(Journey_Details)
            except Exception as e:
                print(f"Error scraping journey: {e}")
            
        return Journeys

    #GETTER METHODS FOR THE JOURNEYS
    def Get_Single_Journeys(self):
        return self.Journeys_Outbound
    
    #def Get_Return_Journeys(self):

if __name__ == "__main__":
    #finder = TicketFinder("London", "Norwich", "17/05/2025",Earliest_Outbound=(19,30),Latest_Outbound=(20,00))
    #finder = TicketFinder("London", "Norwich", "21/05/2025",Earliest_Outbound=(00,30),Latest_Outbound=(23,00),Adults=1,Railcards=["TSU"])
    finder = TicketFinder("London", "Norwich", "22/05/2025", Type="return", Earliest_Inbound=(14,00),Latest_Inbound=(16,00),Return_Date="23/05/2025",Earliest_Outbound=(19,30),Latest_Outbound=(20,00),Adults=2,Railcards=["TSU"])
    single = TicketFinder(
        "Norwich",
        "London",
        "25/05/2025",
        Earliest_Outbound=(14,00),
        Latest_Outbound=(16,00))
    
    # Return journey from Norwich to London for two adults, each with a 16-25 Saver railcard
    return_finder = TicketFinder(
        "Norwich",
        "London",
        "25/05/2025",
        Type="return",
        Return_Date="27/05/2025",
        Earliest_Outbound=(14, 0),
        Latest_Outbound=(16, 0),
        Earliest_Inbound=(20, 0),
        Latest_Inbound=(23, 59),
        Adults=2,
        Railcards=["YNG", "YNG"]  
    )
    
    #single.Search()
    #return_finder.Search()


    #Gets test json data from test.json
    # with open("test.json", "r") as f:
    #     Test_Data = json.load(f)
    # finder.Parse_Journey_Data(Test_Data)We couldn't find any services for the journey you have requested


#TO DO:
#Make the price account for railcards
#COMPLETE - Take railcards as a list of railcards so that multiple can be used if there are more than one passenger
#COMPLETE - If railcard != None, allow for all of the journeys to load on the page (when len of journeys == number of journeys in the requests)
#COMPLETE - Once all journeys loaded, scrape the prices from them

#COMPLETE - Make the price account for number of adults and children

#After scraping is done, return True.
#Then make method Get_Single_Journeys which returns a list of outbound journeys

#If type == return, output the outbound journeys found first, make a method that takes an outbound journey (as its time) and outputs the inbound journeys found

