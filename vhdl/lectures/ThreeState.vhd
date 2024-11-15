library IEEE;
use IEEE.STD_LOGIC_1164.ALL;

entity TriStateBuffer is
  Port (
    Data    : in  std_logic;
    Enable  : in  std_logic;
    Output  : out std_logic
  );
end TriStateBuffer;

architecture Behavioral of TriStateBuffer is
begin
  Output <= Data when Enable = '1' else 'Z';
end Behavioral;

