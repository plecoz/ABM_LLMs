import datetime
import json
import sys
sys.path.append('../../')

class ShortTermMemory:
    def __init__(self):
        # PERSONA HYPERPARAMETERS
    # <vision_r> denotes the number of tiles that the persona can see around 
    # them. 
        self.vision_r = 4
    # <att_bandwidth> TODO 
        self.att_bandwidth = 3
    # <retention> TODO 
        self.retention = 5

    # WORLD INFORMATION
    # Perceived world time. 
        self.curr_time = None
    # Current x,y tile coordinate of the persona. 
        self.curr_tile = None
    # Perceived world daily requirement. 
        self.daily_plan_req = None
    
    # THE CORE IDENTITY OF THE PERSONA 
    # Base information about the persona.
        self.name = None
        self.first_name = None
        self.last_name = None
        self.age = None
        # L0 permanent core traits.  
        self.innate = None
        # L1 stable traits.
        self.learned = None
        # L2 external implementation. 
        self.currently = None
        self.lifestyle = None
        self.living_area = None

        # REFLECTION VARIABLES
        self.concept_forget = 100
        self.daily_reflection_time = 60 * 3
        self.daily_reflection_size = 5
        self.overlap_reflect_th = 2
        self.kw_strg_event_reflect_th = 4
        self.kw_strg_thought_reflect_th = 4

        # New reflection variables
        self.recency_w = 1
        self.relevance_w = 1
        self.importance_w = 1
        self.recency_decay = 0.99
        self.importance_trigger_max = 150
        self.importance_trigger_curr = self.importance_trigger_max
        self.importance_ele_n = 0 
        self.thought_count = 5

        # PERSONA PLANNING 
        # <daily_req> is a list of various goals the persona is aiming to achieve
        # today. 
        # e.g., ['Work on her paintings for her upcoming show', 
        #        'Take a break to watch some TV', 
        #        'Make lunch for herself', 
        #        'Work on her paintings some more', 
        #        'Go to bed early']
        # They have to be renewed at the end of the day, which is why we are
        # keeping track of when they were first generated. 
        self.daily_req = []
        # <f_daily_schedule> denotes a form of long term planning. This lays out 
        # the persona's daily plan. 
        # Note that we take the long term planning and short term decomposition 
        # appoach, which is to say that we first layout hourly schedules and 
        # gradually decompose as we go. 
        # Three things to note in the example below: 
        # 1) See how "sleeping" was not decomposed -- some of the common events 
        #    really, just mainly sleeping, are hard coded to be not decomposable.
        # 2) Some of the elements are starting to be decomposed... More of the 
        #    things will be decomposed as the day goes on (when they are 
        #    decomposed, they leave behind the original hourly action description
        #    in tact).
        # 3) The latter elements are not decomposed. When an event occurs, the
        #    non-decomposed elements go out the window.  
        # e.g., [['sleeping', 360], 
        #         ['wakes up and ... (wakes up and stretches ...)', 5], 
        #         ['wakes up and starts her morning routine (out of bed )', 10],
        #         ...
        #         ['having lunch', 60], 
        #         ['working on her painting', 180], ...]
        self.f_daily_schedule = []
        # <f_daily_schedule_hourly_org> is a replica of f_daily_schedule
        # initially, but retains the original non-decomposed version of the hourly
        # schedule. 
        # e.g., [['sleeping', 360], 
        #        ['wakes up and starts her morning routine', 120],
        #        ['working on her painting', 240], ... ['going to bed', 60]]
        self.f_daily_schedule_hourly_org = []
        
        # CURR ACTION 
        # <address> is literally the string address of where the action is taking 
        # place.  It comes in the form of 
        # "{world}:{sector}:{arena}:{game_objects}". It is important that you 
        # access this without doing negative indexing (e.g., [-1]) because the 
        # latter address elements may not be present in some cases. 
        # e.g., "dolores double studio:double studio:bedroom 1:bed"
        self.act_address = None
        # <start_time> is a python datetime instance that indicates when the 
        # action has started. 
        self.act_start_time = None
        # <duration> is the integer value that indicates the number of minutes an
        # action is meant to last. 
        self.act_duration = None
        # <description> is a string description of the action. 
        self.act_description = None
        # <pronunciatio> is the descriptive expression of the self.description. 
        # Currently, it is implemented as emojis. 
        self.act_pronunciatio = None
        # <event_form> represents the event triple that the persona is currently 
        # engaged in. 
        self.act_event = (self.name, None, None)

        # <obj_description> is a string description of the object action. 
        self.act_obj_description = None
        # <obj_pronunciatio> is the descriptive expression of the object action. 
        # Currently, it is implemented as emojis. 
        self.act_obj_pronunciatio = None
        # <obj_event_form> represents the event triple that the action object is  
        # currently engaged in. 
        self.act_obj_event = (self.name, None, None)

        # <chatting_with> is the string name of the persona that the current 
        # persona is chatting with. None if it does not exist. 
        self.chatting_with = None
        # <chat> is a list of list that saves a conversation between two personas.
        # It comes in the form of: [["Dolores Murphy", "Hi"], 
        #                           ["Maeve Jenson", "Hi"] ...]
        self.chat = None
        # <chatting_with_buffer>  
        # e.g., ["Dolores Murphy"] = self.vision_r
        self.chatting_with_buffer = dict()
        self.chatting_end_time = None

        # <path_set> is True if we've already calculated the path the persona will
        # take to execute this action. That path is stored in the persona's 
        # scratch.planned_path.
        self.act_path_set = False
        # <planned_path> is a list of x y coordinate tuples (tiles) that describe
        # the path the persona is to take to execute the <curr_action>. 
        # The list does not include the persona's current tile, and includes the 
        # destination tile. 
        # e.g., [(50, 10), (49, 10), (48, 10), ...]
        self.planned_path = []
    
def get_str_iss(self): 
    """
    ISS stands for "identity stable set." This describes the commonset summary
    of this persona -- basically, the bare minimum description of the persona
    that gets used in almost all prompts that need to call on the persona. 

    INPUT
      None
    OUTPUT
      the identity stable set summary of the persona in a string form.
    EXAMPLE STR OUTPUT
      "Name: Dolores Heitmiller
       Age: 28
       Innate traits: hard-edged, independent, loyal
       Learned traits: Dolores is a painter who wants live quietly and paint 
         while enjoying her everyday life.
       Currently: Dolores is preparing for her first solo show. She mostly 
         works from home.
       Lifestyle: Dolores goes to bed around 11pm, sleeps for 7 hours, eats 
         dinner around 6pm.
       Daily plan requirement: Dolores is planning to stay at home all day and 
         never go out."
    """
    commonset = ""
    commonset += f"Name: {self.name}\n"
    commonset += f"Age: {self.age}\n"
    commonset += f"Innate traits: {self.innate}\n"
    commonset += f"Learned traits: {self.learned}\n"
    commonset += f"Currently: {self.currently}\n"
    commonset += f"Lifestyle: {self.lifestyle}\n"
    commonset += f"Daily plan requirement: {self.daily_plan_req}\n"
    commonset += f"Current Date: {self.curr_time.strftime('%A %B %d')}\n"
    return commonset


def get_str_name(self): 
        return self.name


def get_str_firstname(self): 
    return self.first_name


def get_str_lastname(self): 
    return self.last_name


def get_str_age(self): 
    return str(self.age)


def get_str_innate(self): 
    return self.innate


def get_str_learned(self): 
    return self.learned


def get_str_currently(self): 
    return self.currently


def get_str_lifestyle(self): 
    return self.lifestyle


def get_str_daily_plan_req(self): 
    return self.daily_plan_req


def get_str_curr_date_str(self): 
    return self.curr_time.strftime("%A %B %d")


def get_curr_event(self):
    if not self.act_address: 
      return (self.name, None, None)
    else: 
      return self.act_event


def get_curr_event_and_desc(self): 
    if not self.act_address: 
      return (self.name, None, None, None)
    else: 
      return (self.act_event[0], 
              self.act_event[1], 
              self.act_event[2],
              self.act_description)


def get_curr_obj_event_and_desc(self): 
    if not self.act_address: 
      return ("", None, None, None)
    else: 
      return (self.act_address, 
              self.act_obj_event[1], 
              self.act_obj_event[2],
              self.act_obj_description)
    


def add_new_action(self, 
                     action_address, 
                     action_duration,
                     action_description,
                     action_pronunciatio, 
                     action_event,
                     chatting_with, 
                     chat, 
                     chatting_with_buffer,
                     chatting_end_time,
                     act_obj_description, 
                     act_obj_pronunciatio, 
                     act_obj_event, 
                     act_start_time=None): 
    self.act_address = action_address
    self.act_duration = action_duration
    self.act_description = action_description
    self.act_pronunciatio = action_pronunciatio
    self.act_event = action_event

    self.chatting_with = chatting_with
    self.chat = chat 
    if chatting_with_buffer: 
      self.chatting_with_buffer.update(chatting_with_buffer)
    self.chatting_end_time = chatting_end_time

    self.act_obj_description = act_obj_description
    self.act_obj_pronunciatio = act_obj_pronunciatio
    self.act_obj_event = act_obj_event
    
    self.act_start_time = self.curr_time
    
    self.act_path_set = False


def act_time_str(self): 
    """
    Returns a string output of the current time. 

    INPUT
      None
    OUTPUT 
      A string output of the current time.
    EXAMPLE STR OUTPUT
      "14:05 P.M."
    """
    return self.act_start_time.strftime("%H:%M %p")


def act_check_finished(self): 
    """
    Checks whether the self.Action instance has finished.  

    INPUT
      curr_datetime: Current time. If current time is later than the action's
                     start time + its duration, then the action has finished. 
    OUTPUT 
      Boolean [True]: Action has finished.
      Boolean [False]: Action has not finished and is still ongoing.
    """
    if not self.act_address: 
      return True
      
    if self.chatting_with: 
      end_time = self.chatting_end_time
    else: 
      x = self.act_start_time
      if x.second != 0: 
        x = x.replace(second=0)
        x = (x + datetime.timedelta(minutes=1))
      end_time = (x + datetime.timedelta(minutes=self.act_duration))

    if end_time.strftime("%H:%M:%S") == self.curr_time.strftime("%H:%M:%S"): 
      return True
    return False


def act_summarize(self):
    """
    Summarize the current action as a dictionary. 

    INPUT
      None
    OUTPUT 
      ret: A human readable summary of the action.
    """
    exp = dict()
    exp["persona"] = self.name
    exp["address"] = self.act_address
    exp["start_datetime"] = self.act_start_time
    exp["duration"] = self.act_duration
    exp["description"] = self.act_description
    exp["pronunciatio"] = self.act_pronunciatio
    return exp


def act_summary_str(self):
    """
    Returns a string summary of the current action. Meant to be 
    human-readable.

    INPUT
      None
    OUTPUT 
      ret: A human readable summary of the action.
    """
    start_datetime_str = self.act_start_time.strftime("%A %B %d -- %H:%M %p")
    ret = f"[{start_datetime_str}]\n"
    ret += f"Activity: {self.name} is {self.act_description}\n"
    ret += f"Address: {self.act_address}\n"
    ret += f"Duration in minutes (e.g., x min): {str(self.act_duration)} min\n"
    return ret


def get_str_daily_schedule_summary(self): 
    ret = ""
    curr_min_sum = 0
    for row in self.f_daily_schedule: 
      curr_min_sum += row[1]
      hour = int(curr_min_sum/60)
      minute = curr_min_sum%60
      ret += f"{hour:02}:{minute:02} || {row[0]}\n"
    return ret


def get_str_daily_schedule_hourly_org_summary(self): 
    ret = ""
    curr_min_sum = 0
    for row in self.f_daily_schedule_hourly_org: 
      curr_min_sum += row[1]
      hour = int(curr_min_sum/60)
      minute = curr_min_sum%60
      ret += f"{hour:02}:{minute:02} || {row[0]}\n"
    return ret
