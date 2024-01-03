#include <stdio.h>

#ifndef __LOG_H__
#define __LOG_H__

// 1、定义几个日志等级，用于打印信息和调试
typedef enum
{
    LOG_LEVEL_ERROR = 1,
    LOG_LEVEL_WARN = 2,
    LOG_LEVEL_INFO = 3
} LOG_LEVEL;

// 暂时手动定义日志等级为 INFO 级别，便于调试
#ifndef LOG_LEV
#define LOG_LEV LOG_LEVEL_INFO
#endif


// 2、定义个 check 的宏来简化开发时代码内部的判断
#ifdef LOG_LEV
#define CHECK(cond) do { \
    if (! (cond)) { \
        fprintf(stderr, "CHECK failed: %s\n", #cond); \
        abort(); \
    } \
} while (0)
#else
// 不做任何操作
#define CHECK(cond)
#endif


// 3、打印调试信息
#ifdef LOG_LEV
#define _Log_Gen(file, func, line, level, levelStr, fmt, ...)                                                                                                     \
    do                                                                                                                                                            \
    {                                                                                                                                                             \
        if (level <= LOG_LEV)                                                                                                                                     \
        {                                                                                                                                                         \
            printf("[%s %s]: LEVEL: %s | located on: %s->%s()->line %d | info: " fmt "\n", \
            __DATE__, __TIME__, levelStr, file, func, line, ##__VA_ARGS__); \
        }                                                                                                                                                         \
    } while (0)
#else
// 不做任何操作
#define _Log_Gen(file, func, line, level, levelStr, fmt, ...)
#endif

#define LOG_ERROR(fmt, ...) _Log_Gen(__FILE__, __FUNCTION__, __LINE__, LOG_LEVEL_ERROR, "LOG_ERR", fmt, ##__VA_ARGS__)
#define LOG_WARN(fmt, ...) _Log_Gen(__FILE__, __FUNCTION__, __LINE__, LOG_LEVEL_WARN, "LOG_WAR", fmt, ##__VA_ARGS__)
#define LOG_INFO(fmt, ...) _Log_Gen(__FILE__, __FUNCTION__, __LINE__, LOG_LEVEL_INFO, "LOG_INFO", fmt, ##__VA_ARGS__)

#endif
