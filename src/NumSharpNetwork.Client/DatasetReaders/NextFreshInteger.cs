using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;

namespace NumSharpNetwork.Client.DatasetReaders
{
    // ensure every index is visited once and only once
    public class NextFreshInteger
    {
        private int[] map;
        private int rightBound;

        private readonly Random random = new Random();

        public NextFreshInteger(int start, int exclusiveEnd)
        {
            map = Enumerable.Range(start, exclusiveEnd - start).ToArray();
            rightBound = exclusiveEnd - start - 1;
        }

        public int GetNextFreshInteger()
        {
            if (rightBound < 0)
            {
                return -1;
            }
            int selectedIndex = this.random.Next(0, rightBound + 1);
            int selected = map[selectedIndex];
            map[selectedIndex] = map[rightBound];
            rightBound--;
            return selected;
        }

        // public IEnumerable<int> GetNextFreshInteger(int start, int exclusiveEnd)
        // {
        //     int[] map = Enumerable.Range(start, exclusiveEnd - start).ToArray();

        //     // inclusive
        //     int rightBound;
        //     for (rightBound = map.Length - 1; rightBound >= 0; rightBound--)
        //     {
        //         int selectedIndex = this.random.Next(0, rightBound + 1);
        //         int selected = map[selectedIndex];
        //         map[selectedIndex] = map[rightBound];
        //         yield return selected;
        //     }
        // }
    }
}
