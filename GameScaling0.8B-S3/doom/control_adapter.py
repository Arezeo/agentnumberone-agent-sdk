"""Transparent ViZDoom mapping for GameScaling packets; no game-playing policy."""
import math
import vizdoom as v
BUTTONS=[v.Button.MOVE_FORWARD,v.Button.MOVE_BACKWARD,v.Button.MOVE_LEFT,v.Button.MOVE_RIGHT,v.Button.ATTACK,v.Button.SPEED,v.Button.USE,v.Button.TURN_LEFT,v.Button.TURN_RIGHT,v.Button.TURN_LEFT_RIGHT_DELTA,v.Button.LOOK_UP_DOWN_DELTA]+[getattr(v.Button,'SELECT_WEAPON'+str(i)) for i in range(1,10)]+[v.Button.SELECT_NEXT_WEAPON,v.Button.SELECT_PREV_WEAPON]
DIGITS={'one':1,'two':2,'three':3,'four':4,'five':5,'six':6,'seven':7,'eight':8,'nine':9}
KEYS={'W':v.Button.MOVE_FORWARD,'Up':v.Button.MOVE_FORWARD,'S':v.Button.MOVE_BACKWARD,'Down':v.Button.MOVE_BACKWARD,'A':v.Button.MOVE_LEFT,'D':v.Button.MOVE_RIGHT,'LB':v.Button.ATTACK,'Ctrl':v.Button.ATTACK,'Shift':v.Button.SPEED,'Space':v.Button.USE,'Left':v.Button.TURN_LEFT,'Right':v.Button.TURN_RIGHT,**{k:getattr(v.Button,'SELECT_WEAPON'+str(n)) for k,n in DIGITS.items()}}
CHUNK_TICS=7

def slot_for_tic(tic):
 # Native 35 Hz engine: 7 tics span 200 ms. Sample six 30 Hz held states.
 if not 0<=tic<CHUNK_TICS:raise ValueError('Packet tic must be in [0,6]')
 return min(5,int(tic*30/35))

def packet_vector(action,tic,width=1280,height=720):
 values={b:0.0 for b in BUTTONS};unsupported=[]
 for token in action.steps[slot_for_tic(tic)]:
  button=KEYS.get(token)
  if button is None:unsupported.append(token)
  else:values[button]=1.0
 if tic==0:
  x,y,z=action.mouse
  # One displacement per packet. FOV-based degree conversion is a local
  # adapter choice: the authors did not publish their mouse sensitivity.
  horizontal_fov=90.0;vertical_fov=math.degrees(2*math.atan(math.tan(math.radians(horizontal_fov/2))*height/width))
  values[v.Button.TURN_LEFT_RIGHT_DELTA]=x/1000*horizontal_fov
  values[v.Button.LOOK_UP_DOWN_DELTA]=-y/1000*vertical_fov
  if z:values[v.Button.SELECT_PREV_WEAPON if z>0 else v.Button.SELECT_NEXT_WEAPON]=1.0
 return [values[b] for b in BUTTONS],unsupported
